import argparse
from pathlib import Path
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Capture a reference image from the desktop webcam.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reference.jpg"),
        help="Destination filename for the captured image.",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Webcam index (0 is default).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1280,
        help="Requested capture width.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=720,
        help="Requested capture height.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Seconds to wait before accepting captures (allows auto-exposure to settle).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    print("Press SPACE to capture, or ESC to exit without saving.")
    start_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed, exiting.")
            break

        cv2.imshow("Reference Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Exit requested, no image saved.")
            break
        if key == 32 and time.perf_counter() - start_time >= args.warmup:  # SPACE
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), frame)
            print(f"Saved reference image to {args.output}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

