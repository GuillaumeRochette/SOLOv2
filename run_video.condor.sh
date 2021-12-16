#! /bin/bash
set -e

src_video=$1
dst_video=$2
labels=${3:-""}

SOLOv2_ROOT=$WRKSPCE/SOLOv2 && \
git clone https://github.com/GuillaumeRochette/SOLOv2.git $SOLOv2_ROOT && \
cd $SOLOv2_ROOT && \
exec python run_video.py \
    --src_video $src_video \
    --dst_video $dst_video \
    --cfg $CFG_SOLOv2_X101_DCN_3x \
    --ckpt $CKPT_SOLOv2_X101_DCN_3x \
    --labels $labels
