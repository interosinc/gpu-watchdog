FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04

# Install Python3
RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get upgrade -y

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    build-essential \
    python3 \
    python3-dev \
    wget \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    automake \
    autoconf \
    pkg-config \
    git \
    libtool \
    curl \
    libpq-dev \
    zlib1g-dev \
    ca-certificates \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Update everything
RUN bash -c "python3.8 -m pip install -U pip"

RUN rm -rf /usr/bin/python3 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python3

RUN python3 -m pip install -U pip setuptools

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app

USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 755 /home/user


ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
COPY requirements.txt ./

ENV PATH="/home/user/.local/bin:${PATH}"
RUN python3 -m pip install -U pip setuptools
RUN python3 -m pip install --no-cache-dir -r ./requirements.txt --use-deprecated=legacy-resolver

# Copy local code to container image
COPY . ./

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python3 gpu_watchdog.py"]
