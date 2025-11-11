# YOLOE Desktop Deployment Toolkit

This workspace contains a streamlined setup for running Ultralytics YOLOE models on a desktop using a standard webcam. It includes scripts to capture prompt references, generate ONNX exports with baked prompts, and run real-time inference or automation demos.

---

## Environment Setup
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics opencv-python onnxruntime onnx
```

The project already caches the prompt-free and promptable YOLOE weights (`yoloe-11s-seg-pf.pt`, `yoloe-11s-seg.pt`) under the repo root. Ultralytics will re-download them automatically if removed.

---

## Key Scripts (`scripts/`)

| Script | Purpose |
| --- | --- |
| `yoloe_run_webcam.py` | Run YOLOE/YOLO models on the desktop webcam with FPS overlay, masks toggle, and headless mode. |
| `export_yoloe_onnx.py` | Convert YOLOE PyTorch weights to ONNX, optionally baking text prompts or image-derived prompts. |
| `capture_reference_image.py` | Capture still images from the webcam for use with image-prompted models. |
| `draw_reference_box.py` | Interactively draw bounding boxes on reference images and emit coordinates. |
| `demo_object_counter.py` | Count target detections and trigger once a threshold is met. |
| `demo_location.py` | Track normalized coordinates of a target class and log when it enters a region. |

All scripts accept `--help` for runtime details.

---

## Typical Workflows

### Prompt-Free Baseline
```bash
.\.venv\Scripts\Activate.ps1
python scripts\yoloe_run_webcam.py --model yoloe-11s-seg-pf.pt --conf 0.25
```

### Text-Prompted Model Export
```bash
python scripts\export_yoloe_onnx.py ^
  --prompts "pink keyboard" "blue mug" ^
  --imgsz 320 ^
  --output models\desk.onnx ^
  --overwrite

python scripts\yoloe_run_webcam.py --model models\desk.onnx --conf 0.2
```

### Image-Prompted Model Export
```bash
# Capture a reference image from the webcam
python scripts\capture_reference_image.py --output data\golem.jpg

# Draw a bounding box around the object of interest
python scripts\draw_reference_box.py --image data\golem.jpg --output-json data\golem_box.json
# Copy the printed bbox values for the export step

# Export ONNX model using the reference image and bounding box
python scripts\export_yoloe_onnx.py ^
  --image-prompt golem data\golem.jpg 120 80 420 520 ^
  --imgsz 320 ^
  --output models\golem.onnx ^
  --overwrite

python scripts\yoloe_run_webcam.py --model models\golem.onnx --conf 0.15
```

### Automation Demos
```bash
# Count detections
python scripts\demo_object_counter.py --model models\desk.onnx --target-object "blue mug" --target-count 2

# Track location
python scripts\demo_location.py --model models\desk.onnx --target-object "pink keyboard" --trigger-x 0.6 --trigger-y 0.6
```

---

## Notes & Tips
- Inference supports either PyTorch (`.pt`) or ONNX models. GPU users can pass `--device cuda:0`.
- Image-prompt exports must process one label at a time; repeat `--image-prompt` for multiple objects.
- ONNX simplification (`--simplify`) requires `onnxsim` (`pip install onnxsim`).
- For production deployments, run `yoloe_run_webcam.py --headless` and implement downstream logic in the demo scripts.
- Additional guidance and background: see `yoloe_model_reference_guide.md`.

---

## Cleanup
Temporary artifacts (captured images, ONNX exports, etc.) are not auto-removed. Manage `data/` and `models/` folders as needed. The `.venv` directory can be deleted and recreated if dependency issues arise.


