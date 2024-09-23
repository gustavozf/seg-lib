FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update -y && \
    apt install -y \
        git \
        libgl1 \
        libglib2.0-0
RUN pip install seg-lib

WORKDIR /home/sam/