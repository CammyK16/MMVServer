# Build the Server
FROM python:3.11-slim AS server_builder
WORKDIR /home
RUN echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes
RUN apt-get update && apt-get install make \
    apt-utils \
    lsb-release \
    apt-transport-https \
    ca-certificates \
    gnupg \
    sqlite3 \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libsqlite3-dev \
    unixodbc-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    zlib1g-dev \
    wget \
    libgl1\
    libsm6 \
    libxext6 \
    libgl1
COPY . /home
RUN python -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt


EXPOSE 44777

FROM python:3.11-slim
WORKDIR /home
COPY --from=server_builder /home/ /home/

ENV APP_ENV=prod APP_ROUTE_PREFIX=""
ENV LD_LIBRARY_PATH="/home/venv/lib/python3.11/site-packages/nvidia/cublas/lib:\
/home/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
/home/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:\
/home/venv/lib/python3.11/site-packages/nvidia/cufft/lib:\
/home/venv/lib/python3.11/site-packages/nvidia/curand/lib:\
/home/venv/lib/python3.11/site-packages/nvidia/cusolver/lib:\
/home/venv/lib/python3.11/site-packages/nvidia/cusparse/lib"

RUN apt-get update && apt-get install -y \
    ca-certificates sqlite3 libsqlite3-dev zlib1g-dev \
    libgl1 libsm6 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN . venv/bin/activate && \
    # For some reason onnxruntime-gpu will not be properly installed unless installed last
    pip install --force-reinstall --no-deps onnxruntime-gpu==1.23.2
# CMD ["sh", "-c", ". venv/bin/activate && uvicorn \"api:app\" --host 0.0.0.0 --port 44777 --log-level warning --loop asyncio"]
