FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace/cetaceans-filtering

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY animal2vec/requirements-torch2.txt /tmp/animal2vec-requirements.txt
RUN python -m pip install --no-cache-dir "pip==24.0" setuptools wheel \
    && python -m pip install --no-cache-dir numpy==1.23.5 cython==3.2.5 \
    && python -m pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/fairseq.git@920a548ca770fb1a951f7f4289b4d3a0c1bc226f" \
    && python -m pip install --no-cache-dir -r /tmp/animal2vec-requirements.txt

COPY . /workspace/cetaceans-filtering

ENV PYTHONPATH=/workspace/cetaceans-filtering

CMD ["python", "-m", "animal2vec.train"]
