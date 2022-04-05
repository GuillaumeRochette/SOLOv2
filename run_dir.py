import os
import sys

SOLO_ROOT = os.environ.get("SOLO_ROOT")
if SOLO_ROOT is not None:
    sys.path.insert(1, os.path.abspath(os.path.dirname(SOLO_ROOT)))

import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

from PIL import Image
import cv2
import torch

from SOLO.mmdet.apis import inference_detector, init_detector

from process import f, g

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--dst_dir", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--labels", nargs="*", default=None, type=int)
    parser.add_argument("--policy", type=str, default="aggregate")
    parser.add_argument("--src_extension", type=str, default="")
    args = parser.parse_args()

    assert args.src_dir.exists()
    assert args.cfg.exists()
    assert args.ckpt.exists()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_detector(str(args.cfg), str(args.ckpt), device=device)

    if not args.labels:
        args.labels = None
    else:
        args.labels = torch.tensor(args.labels, device=device)

    print(args.src_video)
    print(args.dst_video)
    print(args.cfg)
    print(args.ckpt)

    src_images = sorted(args.src_dir.glob(f"*{args.src_extension}"))
    print(src_images)

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    for i, src_image in enumerate(tqdm(src_images)):
        image = cv2.imread(f"{src_image}")
        (result,) = inference_detector(model, image)
        if result is not None:
            masks, labels, scores = result
            masks = f(
                masks=masks,
                labels=labels,
                scores=scores,
                threshold=args.threshold,
                retained_labels=args.labels,
            )
            mask = g(
                masks=masks,
                policy=args.policy
            )
        else:
            h, w, _ = image.shape
            mask = torch.zeros(h, w, dtype=torch.bool)
        mask = Image.fromarray(mask.cpu().numpy())
        mask.save(args.dst_dir / f"{src_image.stem}.png")
