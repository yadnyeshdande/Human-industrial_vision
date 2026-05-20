# =============================================================================
# core/detector.py  –  YOLO multi-class detector  (v3 – class filter + hot-reload)
# =============================================================================
#
# BUG FIX (Bug #1 – model not loading on change):
#   reload(model_name, target_classes) hot-swaps the model without restarting the
#   process.  unload() is called first to free GPU VRAM, then _load() loads the
#   new model.  Called by detection_process on CTRL_RELOAD_SETTINGS when the
#   stored model name differs from the currently loaded one.
#
# FEATURE #1 – configurable target classes:
#   target_classes=[]      → pass classes=None to YOLO → detect ALL classes.
#   target_classes=[0,2]   → pass classes=[0,2] to YOLO → only those IDs.
#   get_class_names()      → returns {class_id: class_name} for the loaded model.
#   This replaces the previous hardcoded PERSON_CLASS_ID = 0 approach.
#
# BACKWARD COMPAT:
#   detect_persons()              still works (strips cls_id, returns 4-tuples).
#   detect_persons_with_scores()  still works (strips cls_id, returns 5-tuples).
#   detect_batch() now returns    6-tuples: (x1, y1, x2, y2, conf, cls_id).
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
    model_name      : filename in models/ (e.g. "yolov8n.pt").
                      Defaults to SETTINGS.yolo_model.
    conf_threshold  : detection confidence.  Defaults to SETTINGS.detection_confidence.
    target_classes  : list of class IDs to detect.
                      []  = detect ALL classes (no filter passed to YOLO).
                      [0] = detect only class 0.
                      Defaults to SETTINGS.target_classes.

    Hot-reload
    ----------
    Call reload(new_model_name, new_target_classes) to swap the model at
    runtime without recreating the object.  The old model is fully unloaded
    (VRAM cleared) before the new one is loaded.
    """

    def __init__(
        self,
        model_name:     Optional[str]       = None,
        conf_threshold: Optional[float]     = None,
        target_classes: Optional[List[int]] = None,
    ):
        from config.loader import SETTINGS

        self.conf_threshold = conf_threshold or SETTINGS.detection_confidence
        self._model_name    = model_name or SETTINGS.yolo_model

        # empty list  = detect all classes (backward-compat with original code)
        if target_classes is not None:
            self.target_classes: List[int] = list(target_classes)
        else:
            self.target_classes = list(SETTINGS.target_classes)

        self.model        = None
        self.device       = "cpu"
        self.model_loaded = False
        self._use_fp16    = False
        self._class_names: Dict[int, str] = {}

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── public API ─────────────────────────────────────────────────────────

    def get_class_names(self) -> Dict[int, str]:
        """Return a copy of {class_id: class_name} for the loaded model."""
        return dict(self._class_names)

    def reload(self, model_name: str, target_classes: List[int]) -> None:
        """
        Hot-swap the model.

        1. Unloads the current model and clears VRAM.
        2. Updates _model_name + target_classes.
        3. Loads the new model.

        Raises RuntimeError if the new model file is not present in models/.
        """
        logger.info(
            f"[HotReload] {self._model_name} → {model_name}  "
            f"classes={target_classes if target_classes else 'ALL'}"
        )
        self.unload()
        self._model_name    = model_name
        self.target_classes = list(target_classes)
        self._load()

    # ── internals ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        # Hard check – refuse to run if model not in models/ directory
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
            # Pass local path directly – ultralytics will NOT attempt a download
            self.model = YOLO(str(local_path))

            # ── capture class names from model metadata ──────────────────
            if hasattr(self.model, "names") and self.model.names:
                self._class_names = dict(self.model.names)
                preview = ", ".join(
                    f"{k}:{v}"
                    for k, v in list(self._class_names.items())[:10]
                )
                suffix = "…" if len(self._class_names) > 10 else ""
                logger.info(
                    f"Model has {len(self._class_names)} classes: {preview}{suffix}"
                )
            else:
                self._class_names = {}
                logger.warning("Model has no class names – labels will be class IDs")

            # ── device selection ─────────────────────────────────────────
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

    # ── detection API ──────────────────────────────────────────────────────

    def detect_persons(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Backward-compat: returns [(x1,y1,x2,y2), ...]."""
        return [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2, *_ in self.detect_persons_with_scores(frame)
        ]

    def detect_persons_with_scores(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Backward-compat: returns [(x1,y1,x2,y2,conf), ...].
        Strips the cls_id that detect_batch now includes.
        """
        if not self.model_loaded or self.model is None:
            return []
        results = self.detect_batch([frame])
        raw = results[0] if results else []
        return [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf, *_ in raw]

    def detect_batch(
        self, frames: List[np.ndarray]
    ) -> List[List[Tuple[int, int, int, int, float, int]]]:
        """
        MICRO-BATCH GPU INFERENCE.

        Returns:
            Per-frame list of (x1, y1, x2, y2, confidence, cls_id) tuples.
            cls_id is the integer YOLO class.
            Length matches len(frames).  Empty list = no detections.

        Class filtering:
            self.target_classes == []  → classes=None  (YOLO detects all)
            self.target_classes != []  → classes=[…]   (YOLO filters server-side)
        """
        if not self.model_loaded or self.model is None or not frames:
            return [[] for _ in frames]

        # None → YOLO detects all classes; a list → YOLO filters to those IDs
        classes_filter: Optional[List[int]] = (
            self.target_classes if self.target_classes else None
        )

        # ── optimised path: pinned-memory tensor batch ───────────────────
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
                    classes=classes_filter,
                    device=self.device,
                    half=False,   # already converted above
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
                            cls_id = int(box.cls[0].cpu().numpy())
                            x1, y1, x2, y2 = xy
                            frame_dets.append(
                                (int(x1), int(y1), int(x2), int(y2), conf, cls_id)
                            )
                        except Exception:
                            continue
                out.append(frame_dets)
            return out

        except Exception as e_opt:
            # ── fallback path: pass raw numpy list ───────────────────────
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

    # ── status ─────────────────────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        return self.model_loaded and self.model is not None

    def unload(self) -> None:
        """Free model from RAM/VRAM.  Safe to call even if already unloaded."""
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
