FROM python:3.10-slim

RUN pip install torch

WORKDIR /app

COPY minimal_transformer ./minimal_transformer
COPY train.py .
COPY test.py .

CMD ["python", "train.py"]