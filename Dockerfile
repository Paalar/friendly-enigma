# Dockerfile
FROM pytorch/pytorch:latest
FROM python:3.8.7-buster
# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    python-dev && \
    rm -rf /var/lib/apt/lists/*



RUN mkdir /app 
# RUN mkdir /app/datasets
COPY pyproject.toml poetry.lock /app/

WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 
RUN apt-get update
RUN apt-get install 'vim' \
    'libsm6'\ 
    'libxext6'  -y

RUN pip install poetry
RUN poetry install

# COPY ./ /app/
RUN poetry install


# ENTRYPOINT ["poetry", "run", "python", "main.py", "mtl"]
# ENTRYPOINT [ "/bin/sh" ]
