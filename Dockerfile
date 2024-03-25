FROM continuumio/miniconda3

RUN mkdir -p M2_Coursework

COPY . /M2_Coursework

WORKDIR /M2_Coursework

RUN apt-get update && apt-get install -y gcc

RUN conda env update -f environment.yml --name M2_CW

RUN apt-get update && apt-get install -y \
    git

RUN echo "conda activate M2_CW" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN git init
RUN pip install pre-commit
RUN pre-commit install
