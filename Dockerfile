FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ARG USE_CHINA_MIRROR=false
ENV USE_CHINA_MIRROR=${USE_CHINA_MIRROR}

ARG APT_MIRROR_URL=http://mirrors.aliyun.com/ubuntu/
ENV APT_MIRROR_URL=${APT_MIRROR_URL}

ARG PYPI_MIRROR_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PYPI_MIRROR_URL=${PYPI_MIRROR_URL}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml src assets conf app.py /app/

RUN set -eo pipefail && \
    if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        echo "deb ${APT_MIRROR_URL} focal main restricted universe" > /etc/apt/sources.list && \
        echo "deb ${APT_MIRROR_URL} focal-updates main restricted universe" >> /etc/apt/sources.list && \
        echo "deb ${APT_MIRROR_URL} focal-security main restricted universe" >> /etc/apt/sources.list; \
    fi && \
    apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN if getent group 1000 &>/dev/null; then \
        group_name=$(getent group 1000 | cut -d: -f1); \
        groupmod -n syaofox $group_name; \
    else \
        groupadd -g 1000 syaofox; \
    fi && \
    if id -u 1000 &>/dev/null; then \
        user_name=$(getent passwd 1000 | cut -d: -f1); \
        usermod -l syaofox -d /home/syaofox -m $user_name; \
        usermod -g 1000 syaofox; \
    else \
        useradd -m -u 1000 -g 1000 -s /bin/bash syaofox; \
    fi

ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

RUN set -eo pipefail && \
    if [ "$USE_CHINA_MIRROR" = "true" ]; then \
        pip config set global.index-url ${PYPI_MIRROR_URL}; \
        pip config set global.trusted-host $(echo ${PYPI_MIRROR_URL} | cut -d/ -f3); \
    fi && \
    pip install --no-cache-dir --break-system-packages --no-build-isolation -e .

RUN chown -R syaofox:syaofox /app

USER syaofox

EXPOSE 8808

CMD ["python", "app.py", "--model-id", "/app/models/huggingface/VoxCPM2", "--zipenhancer-path", "/app/models/modelscope/speech_zipenhancer_ans_multiloss_16k_base", "--sensevoice-path", "/app/models/modelscope/SenseVoiceSmall", "--local-files-only", "--port", "8808"]