import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOE/YOLO models against a desktop webcam feed."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yoloe-11s-seg-pf.pt"),
        help="Path to model weights (.pt or .onnx). Downloads automatically if missing.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index (e.g. 0,1,2) or video path/URL.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override inference resolution (must be multiple of 32). Leave unset for model default.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for visualisation.",
    )
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="Disable bounding box overlay.",
    )
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Enable segmentation masks (if supported by the model).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1280,
        help="Requested capture width for the webcam.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=720,
        help="Requested capture height for the webcam.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device string, e.g. 'cpu', 'cuda:0'.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening an OpenCV display window. Frames are still processed.",
    )
    return parser.parse_args()


def load_model(model_path: Path, device: str):
    if not model_path.exists():
        model_path = Path(str(model_path))  # ensure str conversion for Ultralytics auto-download
    try:
        model = YOLOE(str(model_path))
    except Exception:
        model = YOLO(str(model_path))
    model.to(device)
    return model


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() and len(args.source) == 1 else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {args.source!r}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    model = load_model(args.model, args.device)

    frame_counter = 0
    fps = 0.0
    last_time = time.perf_counter()

    print("Press 'q' in the preview window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: Failed to read frame. Stopping.")
            break

        predict_kwargs = {"imgsz": args.imgsz, "conf": args.conf}
        # Remove None values to avoid overriding defaults
        predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}

        results = model.predict(frame, verbose=False, **predict_kwargs)
        annotated = results[0].plot(
            boxes=not args.no_boxes,
            masks=args.masks,
        )

        frame_counter += 1
        current_time = time.perf_counter()
        elapsed = current_time - last_time
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            last_time = current_time

        if not args.headless:
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (annotated.shape[1] - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLOE Webcam", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

