FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace/cetaceans-filtering

COPY animal2vec/requirements-torch2.txt /tmp/animal2vec-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/animal2vec-requirements.txt

COPY . /workspace/cetaceans-filtering

ENV PYTHONPATH=/workspace/cetaceans-filtering

CMD ["python", "-m", "animal2vec.train"]
