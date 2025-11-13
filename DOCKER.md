# Docker Setup for YOLOE

This project is containerized using Docker with Ubuntu 22.04 base image.

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and start the container:**
   ```bash
   docker-compose up -d --build
   ```

2. **Access the container:**
   ```bash
   docker-compose exec yoloe bash
   ```

3. **Run scripts inside the container:**
   ```bash
   # Example: Export ONNX model
   python scripts/export_yoloe_onnx.py --prompts "Chair" --imgsz 320 --output models/test1.onnx --overwrite
   
   # Example: Run webcam inference (if webcam is accessible)
   python scripts/yoloe_run_webcam.py --model yoloe-11s-seg-pf.pt --conf 0.25
   ```

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the image:**
   ```bash
   docker build -t yoloe:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -it --rm \
     -v ${PWD}:/app \
     -v ${PWD}/models:/app/models \
     -v ${PWD}/data:/app/data \
     yoloe:latest bash
   ```

3. **For webcam access on Linux:**
   ```bash
   docker run -it --rm \
     -v ${PWD}:/app \
     -v ${PWD}/models:/app/models \
     -v ${PWD}/data:/app/data \
     --device=/dev/video0 \
     yoloe:latest bash
   ```

## Git LFS Handling

The entrypoint script automatically:
- Initializes Git LFS
- Checks if model files are pointers
- Pulls actual model files if needed

If models are missing, you can manually pull them:
```bash
git lfs pull
```

## Webcam Access

### Linux
Uncomment the `devices` section in `docker-compose.yml` or use `--device=/dev/video0` flag.

### Windows/Mac
Use `network_mode: host` in docker-compose.yml (Linux hosts only) or pass the webcam through Docker Desktop settings.

For Windows, you may need to use a virtual camera or pass through USB devices via Docker Desktop.

## Volumes

The setup mounts:
- Project root (`/app`) - for development
- `./models` - for exported ONNX models
- `./data` - for reference images and data

## Troubleshooting

1. **Git LFS models not available:**
   - Ensure Git LFS is installed: `git lfs install`
   - Pull models: `git lfs pull`

2. **Webcam not accessible:**
   - Check device permissions
   - Verify device path: `ls -la /dev/video*`
   - On Windows/Mac, use Docker Desktop device passthrough

3. **Permission errors:**
   - Files created in container may have different ownership
   - Use `chown` if needed or run with appropriate user mapping

