# Use an official Python runtime as a parent image
# FROM python:3.12
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12

# Install common Linux utilities including iproute2 for `ss` command
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    build-essential \
    curl \
    wget \
    vim \
    git \
    bash \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV HOST=0.0.0.0
ENV LISTEN_PORT 8000
EXPOSE 8000

# Set the working directory in the container
WORKDIR /app

# Install pipx for managing poetry or pip-tools
RUN pip install --upgrade pip -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Copy the requirements.txt file
COPY requirements.txt ./

# Install dependencies using poetry
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
# Download the spaCy model
RUN python -m spacy download en_core_web_md

RUN pip install --upgrade langchain langchain-openai

# Copy the rest of the application code
COPY . .