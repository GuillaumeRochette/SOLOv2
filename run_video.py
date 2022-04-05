import os
import sys

SOLO_ROOT = os.environ.get("SOLO_ROOT")
if SOLO_ROOT is not None:
    sys.path.insert(1, os.path.abspath(os.path.dirname(SOLO_ROOT)))

import argparse
from pathlib import Path
from time import time
import shutil
import subprocess
from tqdm import tqdm

from PIL import Image
import skvideo.io
import cv2
import torch

from SOLO.mmdet.apis import inference_detector, init_detector

from process import f, g

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_video", type=Path, required=True)
    parser.add_argument("--dst_video", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--labels", nargs="*", type=int, default=None)
    parser.add_argument("--policy", type=int, default="aggregate")
    args = parser.parse_args()

    assert args.src_video.exists()
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

    videogen = skvideo.io.vreader(str(args.src_video))

    tmp_dir = Path("/tmp") / f"TEMP_{time()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(tmp_dir)

    for i, image in enumerate(tqdm(videogen)):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        mask.save(tmp_dir / f"{i:012d}.png")

    if args.dst_video.exists():
        args.dst_video.unlink()

    if not args.dst_video.parent.exists():
        args.dst_video.parent.mkdir(parents=True)

    cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {args.src_video}"
    fps = subprocess.check_output(args=cmd.split(), universal_newlines=True).strip()
    cmd = f"ffmpeg -f image2 -framerate {fps} -i {tmp_dir}/%012d.png -c:v libx264 -crf 0 -pix_fmt yuv420p -y {args.dst_video}"
    video_process = subprocess.run(cmd.split(), check=True)

    shutil.rmtree(tmp_dir)
