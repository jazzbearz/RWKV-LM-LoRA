FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /

RUN mkdir /workspace
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash
RUN apt-get update && apt-get -y install vim wget bash curl libgl1 software-properties-common openssh-server screen ca-certificates libssl-dev wget gnupg python3 python3-pip python-is-python3 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
#RUN wget wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run && sh cuda_11.7.1_515.65.01_linux.run --toolkit --silent && rm cuda_11.7.1_515.65.01_linux.run
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run && sh cuda_11.7.0_515.43.04_linux.run --toolkit --silent && rm cuda_11.7.0_515.43.04_linux.run
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
# RUN wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run && sudo sh cuda_11.7.0_515.43.04_linux.run --toolkit --silent
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 numpy==1.24.1 
# --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install pytorch-lightning==1.9 deepspeed==0.7.0 
RUN pip install -U jupyterlab ipywidgets jupyter-archive transformers
RUN jupyter nbextension enable --py widgetsnbextension
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN mkdir -p /opt/data/model
RUN mkdir -p /opt/data/train
RUN git clone --branch finetune https://github.com/jazzbearz/RWKV-LM-LoRA.git /opt/rwkv
WORKDIR /opt/rwkv

ADD start.sh /
RUN chmod +x /start.sh
CMD [ "/start.sh" ]