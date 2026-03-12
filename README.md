# Industrial Vision Safety System – v2.0 Industrial Architecture

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install PyTorch with CUDA (RTX 3050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Move the downloaded .pt file to models/

# 4. Edit config.yaml – set your camera RTSP URLs

# 5. Launch
python supervisor.py
```

## Architecture

```
supervisor.py  (main process – never crashes)
│
├── camera_process × N     (one per RTSP camera)
│     RTSP → FrameWriter → shared memory
│
├── detection_process × 1  (GPU YOLO worker)
│     shared memory → inference → result_queue + relay_queue
│
├── relay_process × 1      (USB relay hardware)
│     relay_queue → RelayManager → hardware
│
└── gui_process × 1        (PyQt5 – optional)
      shared memory + result_queue → display
```

## Process RAM Limits

| Process     | Limit   |
|-------------|---------|
| camera      | 400 MB  |
| detection   | 1500 MB |
| relay       | 300 MB  |
| gui         | 700 MB  |
| GPU VRAM    | 5200 MB |

Exceeding any limit causes the process to exit; supervisor restarts it.

## Windows 11 Deployment Checklist

- [ ] Set High Performance power plan
- [ ] Disable sleep / hibernate
- [ ] Disable Windows Update auto-restart
- [ ] Assign static IPs to cameras
- [ ] Install CUDA 12.1 + cuDNN
- [ ] Install NVIDIA driver ≥ 525
- [ ] Place YOLO model in `models/`
- [ ] Set RTSP URLs in `config.yaml`
- [ ] Run 72-hour stress test

## File Structure

```
industrial_vision/
├── supervisor.py          ← ENTRY POINT
├── config.yaml            ← main configuration
├── requirements.txt
├── models/                ← YOLO .pt files
├── logs/                  ← per-process rotating logs
├── snapshots/             ← violation images
├── human_boundaries.json  ← zone definitions (auto-managed)
├── app_settings.json      ← runtime settings (auto-managed)
├── processes/
│   ├── camera_process.py
│   ├── detection_process.py
│   ├── relay_process.py
│   └── gui_process.py
├── ipc/
│   ├── messages.py        ← typed message protocol
│   └── frame_store.py     ← shared-memory frame transfer
├── core/
│   ├── detector.py        ← YOLO wrapper (FP16 + no_grad)
│   ├── geometry.py        ← zone math (polygon intersection)
│   ├── reconnect_policy.py
│   └── relay_hardware.py  ← USB relay abstraction
├── config/
│   ├── schema.py          ← Zone / Camera / AppConfig models
│   └── loader.py          ← ConfigManager + AppSettings
├── ui/
│   ├── main_window.py
│   ├── detection_page.py
│   ├── teaching_page.py
│   ├── video_panel.py
│   ├── zone_editor.py
│   └── settings_page.py
└── utils/
    ├── logger.py          ← per-process rotating logger
    ├── resource_guard.py  ← RAM + GPU guardrails
    └── time_utils.py
```
