# Dockerfile for dev container
FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update

COPY requirements.txt requirements.txt
COPY requirements_nf.txt requirements_nf.txt

RUN pip3 install -r requirements.txt

# normalizing flows stuff
RUN pip3 install torch==1.13.1 torchvision==0.14.1

RUN pip3 install torch_geometric==2.3.1
RUN pip3 install torch-scatter==2.1.1

RUN pip3 install -r requirements_nf.txt

RUN pip3 install pgmpy==0.1.19

WORKDIR "/home/workspace"