FROM tensorflow/tensorflow:latest-gpu-py3
LABEL version="0.1"
LABEL description="leveeNet training environment."

# apt install packages
RUN apt-get update && apt-get install -y \
    vim \
    git \
    libsm6 \
    libxrender1 \
    libxext-dev \
    libnetcdf-dev \
    libnetcdff-dev

# install required packages
RUN pip install keras xarray imgaug tqdm netcdf4
WORKDIR /usr/local/lib/python3.6/dist-packages
RUN git clone https://github.com/Unidata/netcdf4-python.git
WORKDIR /usr/local/lib/python3.6/dist-packages/netcdf4-python
RUN python setup.py install

# make worling directory
RUN mkdir -p /opt/analysis
WORKDIR /opt/analysis