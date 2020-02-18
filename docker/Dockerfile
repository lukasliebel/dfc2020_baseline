FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

MAINTAINER Lukas Liebel <lukas.liebel@tum.de>

# Install utils
RUN apt update && apt install -y \
  tmux \
  htop \
  nano
RUN pip install --upgrade pip

# Install required packages
RUN pip install pandas
RUN pip install matplotlib
RUN pip install rasterio
RUN pip install tqdm
RUN pip install tensorboard
RUN pip install sklearn

ENV workdir /root
WORKDIR ${workdir}

CMD bash
