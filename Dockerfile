FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/huggingface \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/ai-toolkit

WORKDIR ${WORKDIR}

# Install Python 3.10 and system deps
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3-pip git \
    build-essential libgl1-mesa-glx libglib2.0-0 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch nightly (CUDA 12.8)
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --timeout 300 --retries 10 --upgrade

# Clone repo, patch requirements, install
RUN git clone https://github.com/ostris/ai-toolkit.git . && \
    sed -i '/^torch==/d' requirements.txt && \
    sed -i '/^torchvision==/d' requirements.txt && \
    sed -i '/^torchaudio==/d' requirements.txt && \
    pip install -r requirements.txt && \
    pip install runpod

# Optional tools
RUN pip install huggingface_hub google-cloud-storage

# Add your scripts
COPY main.py runpod_serverless.py
COPY new.yaml config/main.yaml
COPY test_input.json test_input.json
COPY dataset_downloader.py dataset_downloader.py

CMD ["python", "-u", "runpod_serverless.py"]
