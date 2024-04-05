# base image and labels
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
LABEL AUTHOR="sumukshashidhar"

# environments
ENV SHELL=/bin/bash
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ARG DEBIAN_FRONTEND=noninteractive

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# set working directory
WORKDIR /usr/src/app

# install python and pip
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    apt-get -y install git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils build-essential gcc libffi-dev python3-dev nodejs && \
    apt-get install -y python3.12 && \
    apt-get upgrade -y && \
    apt-get install -y curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install jupyter
RUN pip install --upgrade jupyterlab jupyterlab-code-formatter black isort jupyterlab_materialdarker python-dotenv tabulate kaleido plotly ipywidgets wandb openai anthropic

# cuda focused
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers bitsandbytes datasets sentencepiece scikit-learn peft vllm
RUN pip install --no-deps packaging ninja einops flash-attn xformers trl accelerate bitsandbytes
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# set HF cache
ENV HF_HOME=/huggingface

CMD ["python3", "./src/train.py"]

# EXPOSE 8888

# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]