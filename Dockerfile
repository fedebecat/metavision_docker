FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt -y install python3-pip
RUN apt -y install libcanberra-gtk-module mesa-utils ffmpeg
RUN apt -y install python3.8-dev
RUN python3.8 -m pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
RUN python3.8 -m pip install pip --upgrade
RUN python3.8 -m pip install "opencv-python>=4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy<=1.21" pandas scipy h5py
RUN python3.8 -m pip install jupyter jupyterlab matplotlib "ipywidgets==7.6.5"
RUN apt -y install cmake libboost-program-options-dev libeigen3-dev

RUN echo "deb [arch=amd64 trusted=yes] https://apt.prophesee.ai/dists/public/wohr2Cho/ubuntu focal sdk" > metavision.list
RUN cp metavision.list /etc/apt/sources.list.d
RUN apt update
RUN apt -y install metavision-sdk
RUN apt -y install metavision-sdk-python3.7

RUN python3 -m pip install numba profilehooks "pytorch_lightning==1.5.10" "tqdm==4.63.0" "kornia==0.6.1"
RUN python3 -m pip install llvmlite "pycocotools==2.0.4" "seaborn==0.11.2" "torchmetrics==0.7.2"
