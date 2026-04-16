FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    build-essential\
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY CLIP/requirements.txt /app/CLIP/requirements.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
    -r /app/CLIP/requirements.txt

COPY CLIP /app/CLIP
WORKDIR /app/CLIP

EXPOSE 44778

# CMD ["uvicorn", "clip_api:app", "--host", "0.0.0.0", "--port", "44778", "--log-level", "warning"]