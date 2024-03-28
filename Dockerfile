FROM python:3.12.2

RUN mkdir -p M2_Coursework

COPY . /M2_Coursework

WORKDIR /M2_Coursework

RUN apt-get update && apt-get install -y gcc

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    git

RUN echo "conda activate M2_CW" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN git init
RUN pip install pre-commit
RUN pre-commit install
