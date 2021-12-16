FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt-get -y update
RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev
RUN apt-get autoremove -y && \
    apt-get autoclean -y  && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

RUN conda clean -y --all

ENV WRKSPCE="/workspace"
ENV SOLO_ROOT="$WRKSPCE/SOLO"
RUN git clone https://github.com/WXinlong/SOLO.git $SOLO_ROOT
WORKDIR $SOLO_ROOT
RUN git checkout c7b294a311bfbc59b982b29dc9d12eff42ca0acb

ENV TORCH_CUDA_ARCH_LIST="5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 5.0+PTX 5.2+PTX 5.3+PTX 6.0+PTX 6.1+PTX 6.2+PTX 7.0+PTX 7.2+PTX 7.5+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="/opt/conda"
ENV FORCE_CUDA="1"

RUN pip install --no-cache-dir cython pycocotools tqdm scikit-video
RUN pip install --no-cache-dir -e .

ENV CFG_ROOT=$SOLO_ROOT/configs/solov2
ENV CFG_SOLOv2_X101_DCN_3x=$CFG_ROOT/solov2_x101_dcn_fpn_8gpu_3x.py

ENV CKPT_ROOT=$SOLO_ROOT/checkpoints
ENV CKPT_SOLOv2_X101_DCN_3x=$CKPT_ROOT/SOLOv2_X101_DCN_3x.pth
RUN mkdir -p $CKPT_ROOT && \
    wget https://cloudstor.aarnet.edu.au/plus/s/KV9PevGeV8r4Tzj/download -O $CKPT_SOLOv2_X101_DCN_3x
WORKDIR $WRKSPCE
