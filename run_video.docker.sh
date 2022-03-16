#! /bin/bash
set -e

src_video=$(realpath $1)
dst_video=$2
labels=${3:-""}

src_video_dir=$(dirname $src_video)
dst_video_dir=$(dirname $dst_video)
mkdir -p $dst_video_dir

dst_video=$(realpath $dst_video)
dst_video_dir=$(realpath $dst_video_dir)

cmd="src_video=$src_video && \
dst_video=$dst_video && \
labels=$labels && "
cmd+='SOLOv2_ROOT=$WRKSPCE/SOLOv2 && \
git clone https://github.com/GuillaumeRochette/SOLOv2.git $SOLOv2_ROOT && \
cd $SOLOv2_ROOT && \
exec python run_video.py \
    --src_video $src_video \
    --dst_video $dst_video \
    --cfg $CFG_SOLOv2_X101_DCN_3x \
    --ckpt $CKPT_SOLOv2_X101_DCN_3x \
    --labels $labels'

echo $cmd

exec docker run \
    --gpus 1 \
    -v $src_video_dir:$src_video_dir \
    -v $dst_video_dir:$dst_video_dir \
    --user $(id -u):$(id -g) \
    guillaumerochette/solo:latest \
    /bin/bash -c "$cmd"
