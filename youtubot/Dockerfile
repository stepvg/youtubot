# Install base Python image
FROM python:3.11-slim-bullseye

# Environment for building
ARG HUGGING_FACE_EMBEDDINGS

WORKDIR /usr/local/app

# Python won't create .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Disable python stdout
ENV PYTHONUNBUFFERED 1

# Redis host config
ENV REDIS_HOST redis
ENV REDIS_PORT 6379

# ENV TELEGRAM_BOT_TOKEN ''
# ENV GIGACHAT_CREDENTIALS ''
ENV HUGGING_FACE_EMBEDDINGS $HUGGING_FACE_EMBEDDINGS

# Copy files to the container
COPY . .

# Install dependencies
RUN  pip3 install --no-cache-dir --upgrade pip \
   && pip3 install --no-cache-dir -r requirements.txt

# Download model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$HUGGING_FACE_EMBEDDINGS')" || true

# Flask entrypoint
ENTRYPOINT ["/usr/local/bin/python", "main.py"]

