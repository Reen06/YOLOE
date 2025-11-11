# YOLOE Custom Detection Deployment Reference

Deterministic instruction map derived from `Custom Object Detection Models Without Training | YOLOE & Raspberry Pi - Tutorial Australia`.

---

## 0. Concept Summary
- YOLOE extends YOLO by pairing a vision backbone with promptable embeddings (text or image) so no fine-tuning loop is required.
- Prompt-free weights (`*-pf`) run a 4 800-class default concept head. Prompted ONNX exports embed the requested class list at conversion time.
- Conversion is one-way: changing prompts, resolution, or model size requires rerunning the relevant export script.

---

## 1. Hardware Inventory
| Component                | Notes                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| Raspberry Pi 5          | 4 GB/8 GB/16 GB tested. Pi 4 works but is slow and unvalidated here.   |
| Cooling                 | Use the official active cooler or better.                              |
| Camera                  | Camera Module V3 + Pi 5 FFC adapter (different-width CSI).             |
| Storage & Power         | ≥32 GB microSD, official Pi 27 W PSU.                                  |
| I/O                     | Micro-HDMI→HDMI cable, display, USB keyboard & mouse.                  |

---

## 2. Software Prerequisites
1. Raspberry Pi OS (latest, 64-bit recommended).
2. System updated via `sudo apt update && sudo apt upgrade`.
3. Python 3.11+ (bundled) with:
   - `pip install ultralytics`
   - `pip install opencv-python-headless`
   - `pip install picamera2`
4. Thonny IDE (pre-installed on Pi OS; ensure configured for system interpreter).
5. Download and extract the companion ZIP containing:
   - `YOLOE Run Model.py`
   - `Prompt-Free ONNX Conversion.py`
   - `Text-Prompt ONNX Conversion.py`
   - `Image-Prompt Capture.py`
   - `Image-Prompt Draw Box.py`
   - `Image-Prompt ONNX Conversion.py`
   - `Demo Object Counter.py`
   - `Demo Location.py`

Keep all scripts and exported artifacts in a single project folder to avoid path drift or accidental overwrites.

---

## 3. Camera Bring-Up
1. Power off Pi.
2. Connect Camera Module V3 ribbon:
   - Wide end → camera.
   - Narrow end → Pi 5 CSI slot, contacts facing toward board.
3. Lock both FFC latches; avoid tight bends.
4. Boot Pi, confirm camera enumerates:
   ```bash
   libcamera-hello -t 2000
   ```

---

## 4. Prompt-Free Pipeline (Baselining)
1. Open `YOLOE Run Model.py` in Thonny.
2. Key parameters:
   ```python
   picam2.preview_configuration.main.size = (800, 800)  # capture resolution
   model = YOLO("yoloe-11s-seg-pf.pt")                  # prompt-free PyTorch weights
   ```
3. Execution loop:
   - Acquire frame: `picam2.capture_array()`.
   - Inference: `results = model.predict(frame)`.
   - Visualize: `results[0].plot(boxes=True, masks=False)`.
   - Optional FPS overlay via OpenCV text.
   - Exit gate: `if cv2.waitKey(1) == ord("q")`.
4. First run downloads weights and dependencies; wait for completion. Subsequent runs start immediately.

---

## 5. Model Export Matrix

### 5.1 Format Selection
| Format | Script Source | Pros                                    | Notes                                  |
|--------|---------------|-----------------------------------------|----------------------------------------|
| PyTorch (`.pt`) | Ultralytics hub | Default download | Slowest on Pi; promptable at runtime. |
| ONNX (`.onnx`) | `model.export(format="onnx", ...)` | Reliable on Pi 5 ARM CPU | Prompts baked during export. |
| NCNN (`.ncnn`) | optional | Similar perf to ONNX | Not covered in depth; test if desired. |

Recommendation: default to ONNX for Raspberry Pi deployments.

### 5.2 Resolution Control (`imgsz`)
- Valid values: multiples of 32 between 32 and 640 inclusive.
- Lower values → higher FPS, reduced detection range & accuracy.
- Example settings:
  - `640` (default): max accuracy, lowest FPS.
  - `320`: balanced speed & range.
  - `128`: high FPS, short-range detection only.

### 5.3 Model Size
| Identifier | File                        | Disk Size | Relative Speed | Relative Accuracy |
|------------|----------------------------|-----------|----------------|-------------------|
| `11s`      | `yoloe-11s-seg(-pf).pt`    | ~25 MB    | Fast           | Baseline          |
| `11m`      | `yoloe-11m-seg(-pf).pt`    | ~40 MB    | Medium         | Improved          |
| `11l`      | `yoloe-11l-seg(-pf).pt`    | ~55 MB    | Slow           | Best              |

Tune size based on object complexity vs throughput requirements.

---

## 6. Prompt-Free ONNX Conversion
1. Open `Prompt-Free ONNX Conversion.py`.
2. Configure:
   ```python
   model = YOLO("yoloe-11s-seg-pf.pt")
   model.export(format="onnx", imgsz=640)
   ```
3. Run script; output `yoloe-11s-seg-pf.onnx`.
4. Update `YOLOE Run Model.py`:
   ```python
   model = YOLO("yoloe-11s-seg-pf.onnx")
   ```
5. Validate FPS gains. Iterate on `imgsz` or model size as needed.

---

## 7. Text-Prompted ONNX Conversion
1. Launch `Text-Prompt ONNX Conversion.py`.
2. Input base weights (non-`pf`):
   ```python
   model = YOLOE("yoloe-11s-seg.pt")
   ```
3. Declare prompt list:
   ```python
   names = [
       "tiger",
       "pizza",
       "beard",
       "pokeball",
       "pink keyboard",
       "yellow and purple cup"
   ]
   ```
4. Adjust model size/resolution as required.
5. Optional: set custom filename ending with `-seg.onnx` (e.g., `"tiger-seg.onnx"`).
6. Run script → ONNX export containing fused prompt embeddings.
7. Reference new model in `YOLOE Run Model.py`.

### Prompt Engineering Notes
- Be explicit about colours, materials, shapes (“blue ceramic mug”, “round yellow toy head”).
- Avoid brand-only prompts unless widely visualized.
- If confidence values remain low, scale up model size or refine descriptors.

---

## 8. Image-Prompted ONNX Conversion

### 8.1 Capture or Import Reference
1. `Image-Prompt Capture.py`: set `OUTPUT_FILENAME = "golem.jpg"`, run, press spacebar to save.
2. Alternatively, copy an external image into the project directory (ensure clear focus & lighting).

### 8.2 Annotate Object Region
1. Run `Image-Prompt Draw Box.py`.
2. Configure `IMAGE_PATH = "golem.jpg"`.
3. Drag tight bounding box around target object.
4. Copy printed coordinates `(x1, y1, x2, y2)` (absolute pixel values).

### 8.3 Convert with Embedded Image Prompts
1. Open `Image-Prompt ONNX Conversion.py`.
2. Provide list of prompt dictionaries:
   ```python
   prompts = [
       {
           "image_path": "golem.jpg",
           "bbox": [x1, y1, x2, y2],
           "label": "copper_golem"  # optional metadata for your own reference
       }
   ]
   ```
3. Select model size (consider `11l` for complex visuals) and `imgsz`.
4. Export to e.g. `golem-seg.onnx`.
5. In runtime script:
   ```python
   model = YOLO("golem-seg.onnx")
   ```
6. Detection output labels will appear as `"object 0"`, `"object 1"`, etc. Maintain a manual mapping table in code comments or data structures.

### Reliability Guidance
- Works best on distinctive geometry/texture.
- Re-take reference image if lighting or pose mismatch causes false negatives.
- If multiple objects required, append additional prompt blocks prior to export.

---

## 9. Operational Extensions

### 9.1 Counting Workflow (`Demo Object Counter.py`)
1. Configure:
   ```python
   TARGET_OBJECT = "hand"
   TARGET_COUNT = 1
   CONFIDENCE_THRESHOLD = 0.2
   ```
2. Script iterates detections, filters by label & confidence, aggregates.
3. Customize trigger block inside:
   ```python
   if object_count >= TARGET_COUNT:
       # ADD YOUR CUSTOM ACTION HERE
   ```
   Insert GPIO, messaging, or logging logic.

### 9.2 Localization Workflow (`Demo Location.py`)
1. Set target and threshold.
2. Each detection yields normalized center coordinates `location['x']`, `location['y']` ∈ [0, 1].
3. Example trigger quadrant:
   ```python
   if location['x'] > 0.5 and location['y'] > 0.5:
       # Custom response
   ```
4. Extend to servo control, PTZ camera alignment, etc.

---

## 10. Performance Optimization
- Iterate model size vs FPS: start small, escalate only if accuracy insufficient.
- Reduce `imgsz` carefully; verify maximum detection distance meets requirements.
- Disable overlay masks if not needed to reclaim marginal CPU time.
- Run headless (`cv2.imshow` removed) for production pipelines to reduce overhead.
- Overclock Pi 5 (with adequate cooling) for additional throughput; monitor thermals and stability.
- For heavy workloads, migrate ONNX to Raspberry Pi AI HAT (follow dedicated guide, convert ONNX → HEF).

---

## 11. Troubleshooting Checklist
| Symptom                                   | Remediation Steps                                                           |
|------------------------------------------|------------------------------------------------------------------------------|
| Blank camera feed                         | Re-seat FFC, run `libcamera-hello`, verify `picamera2` import success.      |
| Script stalls on first run                | Allow model download; ensure network access.                                |
| Low confidence on common objects          | Increase model size, refine prompt wording, ensure lighting.                |
| ONNX export overwrites previous model     | Rename output file pre- or post-export (`<name>-seg.onnx`).                 |
| Image-prompt mislabels background         | Redraw tighter bounding box, choose higher-resolution reference, retry.     |
| Excessive false positives in prompt-free  | Switch to prompted model with limited class list.                           |

---

## 12. Deployment Notes
- Maintain version control of exported ONNX files alongside prompt definitions.
- Document prompt lists or image-label mappings in a JSON/YAML manifest for reproducibility.
- Automate periodic Ultralytics updates (`pip install --upgrade ultralytics`) but regression-test exported models afterward.
- For unattended operation, wrap runtime script in a systemd service that handles auto-start and clean shutdown.

---

## 13. Next Actions
1. Select prompt strategy (text or image).
2. Export ONNX with tuned `imgsz` and model size.
3. Validate detection latency & accuracy using baseline scripts.
4. Integrate counting or localization logic into project-specific automation.
5. Iterate configuration parameters until objectives are met.


