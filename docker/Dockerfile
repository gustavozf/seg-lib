FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

COPY seg_lib-1.0.0-py3-none-any.whl source-lib.whl

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update -y && \
    apt install -y \
        git \
        libgl1 \
        libglib2.0-0
RUN pip install source-lib.whl

WORKDIR /home/sam/