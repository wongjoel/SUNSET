# Docker container for solpred

FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

# Make bind targets
RUN mkdir /app \
    && mkdir /data \
    && mkdir /output

# Install Python dependencies from apt
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip

# Get Python packages from pip
WORKDIR /tmp
COPY dependencies.txt ./
RUN pip3 install -r dependencies.txt

# Launch application
CMD ["clj", "-A:cider-clj"]
