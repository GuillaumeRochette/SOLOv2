#! /bin/bash
set -e

src_dir=$1
dst_dir=$2
policy=${3:-"aggregate"}
labels=${4:-""}

SOLOv2_ROOT=$WRKSPCE/SOLOv2 && \
git clone https://github.com/GuillaumeRochette/SOLOv2.git $SOLOv2_ROOT && \
cd $SOLOv2_ROOT && \
exec python run_dir.py \
    --src_dir $src_dir \
    --dst_dir $dst_dir \
    --cfg $CFG_SOLOv2_X101_DCN_3x \
    --ckpt $CKPT_SOLOv2_X101_DCN_3x \
    --policy $policy \
    --labels $labels
