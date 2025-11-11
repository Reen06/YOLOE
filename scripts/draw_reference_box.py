import argparse
import json
from pathlib import Path
from typing import List

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate one or more bounding boxes on an image for YOLOE prompts.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image to annotate.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional list of labels to assign in order. Defaults to object0, object1, ...",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file to persist annotations. Printed to stdout by default.",
    )
    return parser.parse_args()


def build_default_labels(count: int) -> List[str]:
    return [f"object{i}" for i in range(count)]


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Failed to load image {args.image}")

    base = image.copy()
    boxes: list[dict] = []
    drawing = False
    start_point = (-1, -1)
    current_label_index = 0

    labels = args.labels or []

    def mouse_callback(event, x, y, *_):
        nonlocal drawing, start_point, base, image, current_label_index
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            image = base.copy()
            cv2.rectangle(image, start_point, (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end_point = (x, y)
            x1, y1 = start_point
            x2, y2 = end_point
            if x1 == x2 or y1 == y2:
                image = base.copy()
                return
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            current_label = labels[current_label_index] if current_label_index < len(labels) else f"object{len(boxes)}"
            current_label_index += 1
            boxes.append({"label": current_label, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
            base = base.copy()
            cv2.rectangle(base, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                base,
                current_label,
                (x1 + 4, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            image = base.copy()

    cv2.namedWindow("Draw Reference Box")
    cv2.setMouseCallback("Draw Reference Box", mouse_callback)

    print("Instructions:")
    print(" - Left click and drag to draw a bounding box.")
    print(" - Press 'u' to undo the last box.")
    print(" - Press 's' or ENTER to save annotations and exit.")
    print(" - Press 'q' or ESC to quit without saving.")

    while True:
        cv2.imshow("Draw Reference Box", image)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            boxes = []
            print("Exit without saving.")
            break
        if key in (13, ord("s")):  # Enter or s
            break
        if key == ord("u") and boxes:
            boxes.pop()
            base = cv2.imread(str(args.image))
            for box in boxes:
                x1, y1, x2, y2 = box["bbox"]
                cv2.rectangle(base, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    base,
                    box["label"],
                    (x1 + 4, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            image = base.copy()

    cv2.destroyAllWindows()

    if boxes:
        if not labels:
            # ensure labels array lines up
            for idx, box in enumerate(boxes):
                box["label"] = box["label"] or f"object{idx}"
        payload = {"image": str(args.image), "annotations": boxes}
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Saved annotations to {args.output_json}")
        print("Annotations:")
        for box in boxes:
            print(f"{box['label']} {box['bbox'][0]} {box['bbox'][1]} {box['bbox'][2]} {box['bbox'][3]}")


if __name__ == "__main__":
    main()

