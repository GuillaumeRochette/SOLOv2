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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--dst_dir", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--labels", nargs="*", default=None, type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--aggregate", dest="aggregate", action="store_true")
    parser.add_argument("--no-aggregate", dest="aggregate", action="store_false")
    parser.set_defaults(aggregate=True)
    parser.add_argument("--src_extension", type=str, default="")
    args = parser.parse_args()

    assert args.src_dir.exists()
    assert args.cfg.exists()
    assert args.ckpt.exists()

    if not args.labels:
        args.labels = None

    print(args.src_dir)
    print(args.dst_dir)
    print(args.cfg)
    print(args.ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_detector(str(args.cfg), str(args.ckpt), device=device)

    src_images = sorted(args.src_dir.glob(f"*{args.src_extension}"))
    print(src_images)

    if args.dst_dir.exists():
        shutil.rmtree(args.dst_dir)
    args.dst_dir.mkdir(parents=True)

    for i, src_image in enumerate(tqdm(src_images)):
        image = cv2.imread(f"{src_image}")
        (result,) = inference_detector(model, image)
        if result is not None:
            masks, labels, scores = result
            n, h, w = masks.shape
            c = scores > args.threshold
            if args.labels is not None:
                l = torch.zeros_like(c)
                for label in args.labels:
                    l |= labels == label
                c &= l
            masks = masks[c[..., None, None].expand(n, h, w)].reshape(-1, h, w)
            m, h, w = masks.shape
            if m > 0:
                if args.aggregate:
                    mask = masks.sum(dim=-3, dtype=torch.bool)
                else:
                    mask = masks[masks.sum(dim=[-2, -1]).argmax()]
            else:
                mask = torch.zeros(h, w, dtype=torch.bool)
        else:
            h, w, _ = image.shape
            mask = torch.zeros(h, w, dtype=torch.bool)
        mask = Image.fromarray(mask.cpu().numpy())
        mask.save(args.dst_dir / f"{src_image.stem}.png")
