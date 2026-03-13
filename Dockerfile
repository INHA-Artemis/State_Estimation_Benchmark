FROM ros:humble-ros-base-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-venv \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace