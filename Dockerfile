FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y gnupg
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
                    'libsm6'\
                    'libxext6'  -y
RUN apt-get install libglib2.0-0 -y

RUN python3 -m pip install opencv-python
RUN python3 -m pip install tensorboard
RUN python3 -m pip install matplotlib

RUN apt-get install vim -y
RUN apt install git -y

# for MARS
RUN python3 -m pip install faiss-gpu
RUN python3 -m pip install umap-learn
RUN python3 -m pip install joblib
RUN python3 -m pip install Pillow
RUN python3 -m pip install tqdm
RUN python3 -m pip install cmapy
RUN python3 -m pip install ray

# for STEGO
RUN python3 -m pip install hydra-core
RUN python3 -m pip install wget
RUN python3 -m pip install scipy
RUN python3 -m pip install seaborn
RUN python3 -m pip install --ignore-installed PyYAML
RUN python3 -m pip install torchmetrics==0.7.0
RUN python3 -m pip install pytorch-lightning==1.6
RUN python3 -m pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install gcc -y
RUN apt-get install --reinstall build-essential -y

RUN python3 -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

WORKDIR /