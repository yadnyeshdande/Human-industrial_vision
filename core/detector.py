# =============================================================================
# core/detector.py  –  YOLO multi-class detector  (v3)
# =============================================================================
# v2: Model MUST exist locally in models/. No download ever.
#     detect_persons_with_scores() added for enriched bboxes.
#
# v3 changes (Bug #1 + Feature #1):
# -----------------------------------
# FEAT #1 – target_classes parameter:
#   PersonDetector(target_classes=[0, 2]) → only detect classes 0 and 2.
#   PersonDetector(target_classes=[])    → detect ALL classes (default).
#   Replaces the hardcoded classes=[PERSON_CLASS_ID] in detect_batch().
#
# FEAT #1 – get_class_names():
#   Returns {class_id: class_name} for the loaded model.
#   Called by detection_process to label bounding boxes with real names.
#
# BUG #1 – reload() for hot-swap:
#   detector.reload(new_model_name, new_target_classes) unloads the current
#   model (clears GPU VRAM), updates state, and loads the new model.
#   Called by detection_process._drain_control on CTRL_RELOAD_SETTINGS /
#   CTRL_RELOAD_MODEL so no process restart is needed.
#
# detect_batch() return type change:
#   v2 returned [(x1,y1,x2,y2,conf), ...]  – 5-tuples
#   v3 returns  [(x1,y1,x2,y2,conf,cls_id), ...] – 6-tuples
#   detect_persons() and detect_persons_with_scores() are unchanged
#   (they strip cls_id to preserve backward compatibility).
# =============================================================================

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from utils.logger import get_logger

logger = get_logger("Detector")

MODELS_DIR = Path(__file__).parent.parent / "models"


class PersonDetector:
    """
    YOLO-based multi-class detector.

    Parameters
    ----------
    model_name     : filename in models/ (e.g. "yolov8n.pt").
                     Defaults to SETTINGS.yolo_model.
    conf_threshold : detection confidence. Defaults to SETTINGS.detection_confidence.
    target_classes : list of class IDs to detect.
                     []  = detect ALL classes (no filter, backward-compat default).
                     [0] = detect only class 0.
                     Defaults to SETTINGS.target_classes.

    Hot-reload
    ----------
    Call reload(new_model_name, new_target_classes) to swap the model at
    runtime.  The old model is fully unloaded (VRAM cleared) before the
    new one is loaded.  No process restart required.
    """

    # kept for backward compat – still used by _verify_person_class
    PERSON_CLASS_ID   = 0
    PERSON_CLASS_NAME = "person"

    def __init__(
        self,
        model_name:     Optional[str]       = None,
        conf_threshold: Optional[float]     = None,
        target_classes: Optional[List[int]] = None,
    ):
        from config.loader import SETTINGS

        self.conf_threshold = conf_threshold or SETTINGS.detection_confidence
        self._model_name    = model_name or SETTINGS.yolo_model

        # [] = detect all classes (passes classes=None to YOLO)
        if target_classes is not None:
            self.target_classes: List[int] = list(target_classes)
        else:
            self.target_classes = list(SETTINGS.target_classes)

        self.model          = None
        self.device         = "cpu"
        self.model_loaded   = False
        self._use_fp16      = False
        self._class_names:  Dict[int, str] = {}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def get_class_names(self) -> Dict[int, str]:
        """Return a copy of {class_id: class_name} for the loaded model."""
        return dict(self._class_names)

    def reload(self, model_name: str, target_classes: List[int]) -> None:
        """
        Hot-swap the model in-place.

        1. Unloads current model and clears GPU VRAM.
        2. Updates _model_name + target_classes.
        3. Loads the new model.

        Raises RuntimeError if the new model file is not in models/.
        """
        logger.info(
            f"[HotReload] {self._model_name!r} → {model_name!r}  "
            f"classes={target_classes if target_classes else 'ALL'}"
        )
        self.unload()
        self._model_name    = model_name
        self.target_classes = list(target_classes)
        self._load()

    # ── internals ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        # Hard check – refuse to run if model not local
        local_path = MODELS_DIR / self._model_name
        if not local_path.exists():
            msg = (
                f"STARTUP ABORTED: Model file not found: {local_path}\n"
                f"Place the model file in the models/ directory before starting."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            from ultralytics import YOLO
            import torch

            logger.info(f"Loading model: {local_path}")
            # Pass local path directly – ultralytics will NOT download
            self.model = YOLO(str(local_path))

            # Capture class names from model metadata (FEAT #1)
            if hasattr(self.model, "names") and self.model.names:
                self._class_names = dict(self.model.names)
                preview = ", ".join(
                    f"{k}:{v}" for k, v in list(self._class_names.items())[:8]
                )
                suffix = "…" if len(self._class_names) > 8 else ""
                logger.info(
                    f"Model has {len(self._class_names)} classes: {preview}{suffix}"
                )
            else:
                self._class_names = {}
                logger.warning("Model has no class names – labels will be class IDs")

            if torch.cuda.is_available():
                self.device    = "cuda"
                self._use_fp16 = True
                logger.info("YOLO using GPU (CUDA) FP16")
            else:
                logger.info("YOLO using CPU")

            self._warmup()
            self.model_loaded = True

            tc_str = (
                "ALL classes"
                if not self.target_classes
                else ", ".join(
                    f"{c}:{self._class_names.get(c, str(c))}"
                    for c in self.target_classes
                )
            )
            logger.info(f"Detector ready: {self._model_name}  [{tc_str}]")

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Detector load failed: {e}", exc_info=True)
            self.model_loaded = False
            raise

    def _warmup(self) -> None:
        try:
            import torch
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            with torch.no_grad():
                self.model(dummy, device=self.device,
                           verbose=False, half=self._use_fp16)
            if self.device == "cuda":
                torch.cuda.synchronize()
            logger.info("Detector warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-fatal): {e}")

    # ── detection API ─────────────────────────────────────────────────────────

    def detect_persons(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Returns [(x1,y1,x2,y2), ...] – legacy interface, unchanged."""
        return [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2, *_ in self.detect_persons_with_scores(frame)
        ]

    def detect_persons_with_scores(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns [(x1, y1, x2, y2, confidence), ...] – backward-compat.
        Strips cls_id from the v3 6-tuple that detect_batch now returns.
        """
        if not self.model_loaded or self.model is None:
            return []
        results = self.detect_batch([frame])
        raw = results[0] if results else []
        # strip cls_id (element [5]) to preserve 5-tuple contract
        return [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf, *_ in raw]

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Tuple[int, int, int, int, float, int]]]:
        """
        MICRO-BATCH GPU INFERENCE.

        Returns:
            Per-frame list of (x1, y1, x2, y2, confidence, cls_id) 6-tuples.
            Length matches len(frames).  Empty list = no detections.

        Class filtering (FEAT #1):
            target_classes == []  → classes=None   (YOLO detects all)
            target_classes != []  → classes=[…]    (YOLO filters server-side)
        """
        if not self.model_loaded or self.model is None or not frames:
            return [[] for _ in frames]

        # None → YOLO detects all; list → YOLO filters to those IDs
        classes_filter: Optional[List[int]] = (
            self.target_classes if self.target_classes else None
        )

        # Optimised path: pinned-memory → non_blocking device transfer
        try:
            import torch

            cpu_tensors = []
            for f in frames:
                arr = np.ascontiguousarray(f)
                t   = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)
                t   = t.div(255.0)
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                cpu_tensors.append(t)

            batch_tensor = torch.stack(cpu_tensors, dim=0)
            if self.device == "cuda":
                batch_tensor = batch_tensor.to("cuda", non_blocking=True)
                if self._use_fp16:
                    batch_tensor = batch_tensor.half()

            with torch.no_grad():
                results = self.model(
                    batch_tensor,
                    conf=self.conf_threshold,
                    classes=classes_filter,   # FEAT #1: dynamic filter
                    device=self.device,
                    half=False,               # already converted above
                    verbose=False,
                )
                if self.device == "cuda":
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass

            out: List[List[Tuple[int, int, int, int, float, int]]] = []
            for res in results:
                frame_dets: List[Tuple[int, int, int, int, float, int]] = []
                if res.boxes is not None:
                    for box in res.boxes:
                        try:
                            xy     = box.xyxy[0].cpu().numpy()
                            conf   = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())   # FEAT #1
                            x1, y1, x2, y2 = xy
                            frame_dets.append(
                                (int(x1), int(y1), int(x2), int(y2), conf, cls_id)
                            )
                        except Exception:
                            continue
                out.append(frame_dets)
            return out

        except Exception as e_opt:
            # Fallback: pass numpy list directly
            try:
                import torch
                with torch.no_grad():
                    results = self.model(
                        frames,
                        conf=self.conf_threshold,
                        classes=classes_filter,
                        device=self.device,
                        half=self._use_fp16,
                        verbose=False,
                    )
                out = []
                for res in results:
                    frame_dets = []
                    if res.boxes is not None:
                        for box in res.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf   = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            frame_dets.append(
                                (int(x1), int(y1), int(x2), int(y2), conf, cls_id)
                            )
                    out.append(frame_dets)
                return out
            except Exception as e:
                logger.error(
                    f"Batch inference failed (both paths): {e_opt} | {e}"
                )
                return [[] for _ in frames]

    # ── status ────────────────────────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        return self.model_loaded and self.model is not None

    def unload(self) -> None:
        if self.model is not None:
            try:
                del self.model
                self.model        = None
                self.model_loaded = False
                self._class_names = {}
                if self.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info(
                    f"Model '{self._model_name}' unloaded – GPU VRAM cache cleared"
                )
            except Exception as e:
                logger.warning(f"Unload error: {e}")