# =============================================================================
# utils/resource_guard.py  –  RAM / GPU resource guardrails  (v4)
# =============================================================================
# FIX #6: None-guard on all threshold comparisons – no more
#         "'>' not supported between float and NoneType"
# =============================================================================

import os
import time
from typing import Optional
from utils.logger import get_logger

logger = get_logger("ResourceGuard")

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    logger.warning("psutil not installed – RAM monitoring disabled")

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False

# FIX #1 (v3): corrected per-process RAM limits (MB)
RAM_LIMIT_CAMERA    = 600
RAM_LIMIT_DETECTION = 3500
RAM_LIMIT_RELAY     = 512
RAM_LIMIT_GUI       = 1000
VRAM_LIMIT_MB       = 5200

GPU_TEMP_WARNING_C  = 80
GPU_TEMP_CRITICAL_C = 90


class ResourceLimitExceeded(Exception):
    pass


class ResourceGuard:
    """
    Per-process RSS RAM + optional VRAM guard.

    FIX #6: All limit comparisons are guarded:
      if self.ram_limit_mb is None → skip RSS check
      if self.vram_limit_mb is None → skip VRAM check
    Prevents crash when supervisor passes None for optional limits.
    """

    def __init__(
        self,
        ram_limit_mb:     Optional[float],
        vram_limit_mb:    Optional[float] = None,
        check_interval_s: float           = 30.0,
    ):
        # FIX #6: accept None gracefully; fall back to safe defaults
        self.ram_limit_mb   = float(ram_limit_mb) if ram_limit_mb is not None else None
        self.vram_limit_mb  = float(vram_limit_mb) if vram_limit_mb is not None else None
        self.check_interval = check_interval_s
        self._last_check    = 0.0
        self._pid           = os.getpid()
        self._proc          = psutil.Process(self._pid) if _PSUTIL else None
        self._gpu_handle    = None
        if _NVML:
            try:
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    def check(self) -> None:
        now = time.monotonic()
        if now - self._last_check < self.check_interval:
            return
        self._last_check = now

        # ── RSS RAM ───────────────────────────────────────────────────────────
        if self.ram_limit_mb is None:
            pass   # FIX #6: no limit configured → skip
        elif _PSUTIL and self._proc:
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                if rss_mb > self.ram_limit_mb:
                    raise ResourceLimitExceeded(
                        f"RSS RAM {rss_mb:.0f} MB > limit {self.ram_limit_mb:.0f} MB"
                    )
                logger.debug(f"RSS RAM: {rss_mb:.0f}/{self.ram_limit_mb:.0f} MB")
            except ResourceLimitExceeded:
                raise
            except Exception as e:
                logger.warning(f"RAM check error: {e}")

        # ── VRAM ──────────────────────────────────────────────────────────────
        if self.vram_limit_mb is None:
            pass   # FIX #6: no limit → skip
        elif _TORCH and torch.cuda.is_available():
            try:
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                if reserved_mb > self.vram_limit_mb:
                    raise ResourceLimitExceeded(
                        f"VRAM {reserved_mb:.0f} MB > limit {self.vram_limit_mb:.0f} MB"
                    )
                logger.debug(f"VRAM: {reserved_mb:.0f}/{self.vram_limit_mb:.0f} MB")
            except ResourceLimitExceeded:
                raise
            except Exception as e:
                logger.warning(f"VRAM check error: {e}")

    def get_ram_mb(self) -> float:
        if _PSUTIL and self._proc:
            try:
                return self._proc.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    def get_vram_mb(self) -> float:
        # Prefer NVML device-level used VRAM if available (accurate for whole GPU)
        if _NVML and self._gpu_handle:
            try:
                info = _pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                return float(info.used) / (1024 * 1024)
            except Exception:
                pass
        # Fall back to PyTorch per-process allocation/reserved if available
        if _TORCH and torch.cuda.is_available():
            try:
                # use allocated() which reflects actual tensors, reserved() can be misleading
                return torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                try:
                    return torch.cuda.memory_reserved() / (1024 * 1024)
                except Exception:
                    pass
        return 0.0

    def get_gpu_utilization(self) -> float:
        if _NVML and self._gpu_handle:
            try:
                return float(_pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu)
            except Exception:
                pass
        return 0.0

    def get_gpu_temp(self) -> float:
        if _NVML and self._gpu_handle:
            try:
                return float(_pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, _pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
        return 0.0

    def is_gpu_overheating(self) -> bool:
        return self.get_gpu_temp() >= GPU_TEMP_CRITICAL_C

    def gpu_health_summary(self) -> dict:
        return {
            "vram_mb":     self.get_vram_mb(),
            "gpu_util":    self.get_gpu_utilization(),
            "gpu_temp_c":  self.get_gpu_temp(),
            "overheating": self.is_gpu_overheating(),
        }
