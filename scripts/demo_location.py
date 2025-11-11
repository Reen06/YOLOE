import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE


def parse_args():
    parser = argparse.ArgumentParser(description="Track the on-screen location of an object using YOLOE and a webcam.")
    parser.add_argument("--model", type=Path, default=Path("yoloe-11s-seg-pf.pt"), help="Model weights to use.")
    parser.add_argument("--target-object", type=str, default="hand", help="Class name to monitor.")
    parser.add_argument("--confidence-threshold", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--trigger-x", type=float, default=0.5, help="Normalized X threshold for trigger condition.")
    parser.add_argument("--trigger-y", type=float, default=0.5, help="Normalized Y threshold for trigger condition.")
    parser.add_argument("--source", type=int, default=0, help="Webcam index or video file.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size override.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device spec, e.g. 'cpu', 'cuda:0'.")
    parser.add_argument("--frame-width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--frame-height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
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
        highest_conf = None
        location = None

        if boxes is not None and boxes.xywhn is not None:
            for cls, conf, xywhn in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()):
                label = names.get(int(cls), f"class_{int(cls)}")
                if label.lower() == args.target_object.lower() and conf >= args.confidence_threshold:
                    if highest_conf is None or conf > highest_conf:
                        highest_conf = conf
                        location = {"x": xywhn[0], "y": xywhn[1], "width": xywhn[2], "height": xywhn[3]}

        if not args.headless:
            annotated = result.plot()
            if location and highest_conf:
                x_px = int(location["x"] * annotated.shape[1])
                y_px = int(location["y"] * annotated.shape[0])
                cv2.drawMarker(
                    annotated,
                    (x_px, y_px),
                    (0, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )
                cv2.putText(
                    annotated,
                    f"{args.target_object} ({location['x']:.2f}, {location['y']:.2f}) conf={highest_conf:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("Object Location", annotated)

        if location and highest_conf:
            if location["x"] > args.trigger_x and location["y"] > args.trigger_y:
                print(
                    f"HIGH CONFIDENCE {args.target_object} detected at "
                    f"x={location['x']:.3f}, y={location['y']:.3f}, conf={highest_conf:.3f}"
                )

        if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

