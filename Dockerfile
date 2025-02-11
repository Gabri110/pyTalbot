# Dockerfile for Running an OpenMP Environment
FROM ubuntu:22.04

RUN apt-get --yes -qq update \
 && apt-get --yes -qq upgrade \
 && apt-get --yes -qq install \
                      bzip2 \
                      cmake \
                      cpio \
                      curl \
                      g++ \
                      gcc \
                      gfortran \
                      git \
                      gosu \
                      libblas-dev \
                      liblapack-dev \
                      libgsl-dev \
                      libopenmpi-dev \
                      openmpi-bin \
                      python3-dev \
                      python3-pip \
                      virtualenv \
                      wget \
                      zlib1g-dev \
                      vim       \
                      htop      \
                      ffmpeg \
 && apt-get --yes -qq clean \
 && rm -rf /var/lib/apt/lists/*

# Make sure you copy the requirements.txt to the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip3 install -r /app/requirements.txt

CMD [ "/bin/bash" ]