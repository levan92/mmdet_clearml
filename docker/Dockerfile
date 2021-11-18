# ARG PYTORCH="1.9.0"
# ARG CUDA="10.2"
# ARG CUDNN="7"

# ARG PYTORCH="1.9.0"
# ARG CUDA="10.2"
# ARG CUDNN="7"

ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV FORCE_CUDA="1"

# Install MMCV
# RUN pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu113/torch${PYTORCH}/index.html
RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
RUN cd /mmcv && MMCV_WITH_OPS=1 pip install -e .

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir future tensorboard
RUN pip install --no-cache-dir clearml boto3 wget