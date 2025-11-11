import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from ultralytics import YOLOE


@dataclass
class ImagePrompt:
    label: str
    image_path: Path
    bbox: list[float]  # [x1, y1, x2, y2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLOE models to ONNX with optional text or image prompts."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Path to the starting PyTorch weights (.pt). Defaults depend on prompt mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination ONNX path. Defaults to base-model name with '.onnx'.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference resolution for export (must be multiple of 32).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Export format, e.g. onnx, ncnn. Only ONNX is validated here.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run export on, e.g. 'cpu', 'cuda:0'.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Space-separated list of text prompts to bake into the model.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument(
        "--image-prompt",
        action="append",
        nargs=6,
        metavar=("LABEL", "IMAGE", "X1", "Y1", "X2", "Y2"),
        help="Add an image prompt. Repeat flag per object. Coordinates are in pixels.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 where supported.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run ONNX graph simplification after export (requires onnx-simplifier).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Override ONNX opset version.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args()


def collect_text_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompts:
        prompts.extend(p.strip() for p in args.prompts if p.strip())
    if args.prompts_file:
        content = args.prompts_file.read_text(encoding="utf-8")
        prompts.extend(line.strip() for line in content.splitlines() if line.strip())
    # De-duplicate but retain order
    seen = set()
    unique_prompts = []
    for prompt in prompts:
        if prompt not in seen:
            unique_prompts.append(prompt)
            seen.add(prompt)
    return unique_prompts


def collect_image_prompts(arg_values: Iterable[list[str]]) -> list[ImagePrompt]:
    prompts: list[ImagePrompt] = []
    if not arg_values:
        return prompts
    for entry in arg_values:
        label, image_path, x1, y1, x2, y2 = entry
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        prompts.append(ImagePrompt(label=label, image_path=Path(image_path), bbox=bbox))
    return prompts


def ensure_base_model(args: argparse.Namespace, requires_prompt_capable: bool) -> Path:
    if args.base_model:
        return args.base_model
    # Default selection
    return Path("yoloe-11s-seg.pt" if requires_prompt_capable else "yoloe-11s-seg-pf.pt")


def run_text_prompt_export(model: YOLOE, prompts: list[str]) -> None:
    if not prompts:
        raise ValueError("Text prompt export requested without any prompts.")
    model.set_classes(prompts)


def run_image_prompt_export(model: YOLOE, prompts: list[ImagePrompt], imgsz: int, device: str) -> None:
    if not prompts:
        raise ValueError("Image prompt export requested without any prompts.")
    embeddings = []
    labels: list[str] = []
    for idx, prompt in enumerate(prompts):
        if not prompt.image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {prompt.image_path}")
        visual_prompts = {"bboxes": [prompt.bbox], "cls": [0]}
        model.predict(
            source=None,
            visual_prompts=visual_prompts,
            refer_image=str(prompt.image_path),
            imgsz=imgsz,
            device=device,
        )
        current = getattr(model.model, "pe", None)
        if current is None:
            raise RuntimeError(f"Failed to derive embeddings for image prompt #{idx+1} ({prompt.label}).")
        embeddings.append(current.detach().clone())
        labels.append(prompt.label)

    merged = torch.cat(embeddings, dim=1)
    model.set_classes(labels, embeddings=merged)


def main():
    args = parse_args()

    text_prompts = collect_text_prompts(args)
    image_prompts = collect_image_prompts(args.image_prompt or [])
    prompt_modes = sum(bool(x) for x in (text_prompts, image_prompts))
    requires_prompt_capable = prompt_modes > 0

    base_model_path = ensure_base_model(args, requires_prompt_capable)
    if not base_model_path.exists():
        base_model_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLOE(str(base_model_path))
    model.to(args.device)

    if text_prompts and image_prompts:
        raise ValueError("Please choose either text prompts OR image prompts for export, not both simultaneously.")

    if text_prompts:
        run_text_prompt_export(model, text_prompts)
    elif image_prompts:
        run_image_prompt_export(model, image_prompts, args.imgsz, args.device)

    output_path = args.output
    if output_path is None:
        stem = base_model_path.with_suffix("")
        suffix = "-custom" if prompt_modes else ""
        output_path = stem.parent / f"{stem.name}{suffix}.{args.format}"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "simplify": args.simplify,
    }
    if args.opset:
        export_kwargs["opset"] = args.opset

    exported = model.export(**export_kwargs)
    exported_path = Path(exported)
    if output_path != exported_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(output_path)
    print(f"Export complete: {output_path}")


if __name__ == "__main__":
    main()

