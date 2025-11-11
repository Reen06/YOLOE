import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE


def parse_args():
    parser = argparse.ArgumentParser(description="Count occurrences of a target object using a YOLOE model and webcam.")
    parser.add_argument("--model", type=Path, default=Path("yoloe-11s-seg-pf.pt"), help="Model weights to load.")
    parser.add_argument("--target-object", type=str, default="hand", help="Name of the object to count.")
    parser.add_argument("--target-count", type=int, default=1, help="Trigger once this many objects are detected.")
    parser.add_argument("--confidence-threshold", type=float, default=0.2, help="Minimum confidence to consider.")
    parser.add_argument("--source", type=int, default=0, help="Webcam index or video source.")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size override.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string, e.g. 'cpu' or 'cuda:0'.")
    parser.add_argument("--frame-width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--frame-height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--headless", action="store_true", help="Disable the OpenCV display window.")
    return parser.parse_args()


def load_model(path: Path, device: str):
    try:
        model = YOLOE(str(path))
    except Exception:
        model = YOLO(str(path))
    model.to(device)
    return model


def main():
    args = parse_args()
    model = load_model(args.model, args.device)
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    target_met = False
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame capture failed, stopping.")
            break

        predict_kwargs = {"imgsz": args.imgsz, "conf": args.confidence_threshold}
        predict_kwargs = {k: v for k, v in predict_kwargs.items() if v is not None}
        results = model.predict(frame, verbose=False, **predict_kwargs)
        result = results[0]

        names = result.names if isinstance(result.names, dict) else {i: n for i, n in enumerate(result.names)}
        boxes = result.boxes
        confident_targets = []
        if boxes is not None:
            for cls, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                label = names.get(int(cls), f"class_{int(cls)}")
                if label.lower() == args.target_object.lower() and conf >= args.confidence_threshold:
                    confident_targets.append({"confidence": conf, "bbox": xyxy})

        if confident_targets and not args.headless:
            annotated = result.plot()
            cv2.imshow("Object Counter", annotated)
        elif not args.headless:
            cv2.imshow("Object Counter", frame)

        if len(confident_targets) >= args.target_count and not target_met:
            target_met = True
            print(
                f"Target met! Detected {len(confident_targets)} >= {args.target_count} '{args.target_object}' "
                f"with confidence ≥ {args.confidence_threshold}"
            )
            for idx, obj in enumerate(confident_targets, start=1):
                print(f"  #{idx}: confidence={obj['confidence']:.3f} bbox={obj['bbox']}")

        if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

