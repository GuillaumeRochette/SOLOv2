#! /bin/bash
set -e

src_dir=$(realpath $1)
dst_dir=$2
policy=${3:-"aggregate"}
labels=${4:-""}

src_dir_dir=$(dirname $src_dir)
dst_dir_dir=$(dirname $dst_dir)
mkdir -p $dst_dir_dir

dst_dir=$(realpath $dst_dir)
dst_dir_dir=$(realpath $dst_dir_dir)

cmd="src_dir=$src_dir && \
dst_dir=$dst_dir && \
policy=$policy && \
labels=$labels && "
cmd+='SOLOv2_ROOT=$WRKSPCE/SOLOv2 && \
git clone https://github.com/GuillaumeRochette/SOLOv2.git $SOLOv2_ROOT && \
cd $SOLOv2_ROOT && \
exec python run_dir.py \
    --src_dir $src_dir \
    --dst_dir $dst_dir \
    --cfg $CFG_SOLOv2_X101_DCN_3x \
    --ckpt $CKPT_SOLOv2_X101_DCN_3x \
    --policy $policy \
    --labels $labels'

echo $cmd

exec docker run \
    --gpus 1 \
    -v $src_dir_dir:$src_dir_dir \
    -v $dst_dir_dir:$dst_dir_dir \
    --user $(id -u):$(id -g) \
    guillaumerochette/solo:latest \
    /bin/bash -c "$cmd"
