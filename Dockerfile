#inspirate in Sri Raj Paul https://srirajpaul.blogspot.com/2019/07/cuda-gpu-simulator-container.html

FROM ubuntu:18.04

LABEL maintainer="Andreeo"

COPY script.sh /tmp/script.sh

RUN /bin/bash -x /tmp/script.sh
