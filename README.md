SOLOv2
===
---
### Requirements

- If you have a GPU (which is recommended), its CUDA Compute Capability must be >= 5.0 and <=8.0, otherwise this might not work.

### Recommended: Within a Docker Container

#### Install

1. Install Docker and follow post-installation steps: https://docs.docker.com/engine/install/
2. Install NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker

#### Run

- For a video:

```bash
bash run_video.docker.sh /path/to/input/video.mp4 /path/to/output/masks.mp4 "0"
```
- For a directory of images:

```bash
bash run_dir.docker.sh /path/to/image/dir /path/to/mask/dir "0"
```

### Alternative: Run within your Local Environment

#### Install

1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Execute:

```bash
conda create -y -n solo-env python=3.7 pip
conda activate solo-env
conda install -y pytorch=1.3.0 torchvision=0.4.1 cudatoolkit=10.1 -c pytorch

git clone https://github.com/WXinlong/SOLO.git SOLO
cd SOLO
git checkout c7b294a311bfbc59b982b29dc9d12eff42ca0acb
pip install cython pycocotools tqdm scikit-video
pip install -e .
mkdir checkpoints
wget https://cloudstor.aarnet.edu.au/plus/s/KV9PevGeV8r4Tzj/download -O checkpoints/SOLOv2_X101_DCN_3x.pth
cd ..
```

#### Run

- For a video:
```bash
run_video.py --src_video SRC_VIDEO --dst_video DST_VIDEO --cfg CFG --ckpt CKPT
            [--labels LABELS] [--threshold THRESHOLD] [--aggregate] [--no-aggregate]
```
- For a directory of images:
```bash
run_dir.py --src_dir SRC_DIR --dst_dir DST_DIR --cfg CFG --ckpt CKPT [--labels [LABELS]
          [--threshold THRESHOLD] [--aggregate] [--no-aggregate] [--src_extension SRC_EXTENSION]
```