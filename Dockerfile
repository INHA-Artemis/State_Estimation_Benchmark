FROM ros:humble-ros-base-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/app \
    INVARIANT_EKF_RUNNER=/app/compare_repos/invariant-ekf/inekf/build/bin/kaist_vio_runner \
    DRIFT_RUNNER=/app/compare_repos/drift/build/kaist_vio_runner \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        pkg-config \
        python3-dev \
        python3-venv \
        ffmpeg \
        libboost-test-dev \
        libeigen3-dev \
        libspatialindex-dev \
        libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN pip install --no-cache-dir \
        -e /app/compare_repos/filterpy \
        -e /app/compare_repos/Stone-Soup \
    && cmake -S /app/compare_repos/invariant-ekf/inekf \
        -B /app/compare_repos/invariant-ekf/inekf/build \
        -DBUILD_TESTS=OFF \
    && cmake --build /app/compare_repos/invariant-ekf/inekf/build \
        --target kaist_vio_runner \
        --parallel \
    && cmake -S /app/compare_repos/drift \
        -B /app/compare_repos/drift/build \
        -DBUILD_TESTS=OFF \
        -DBUILD_DOC=OFF \
    && cmake --build /app/compare_repos/drift/build \
        --target kaist_vio_runner \
        --parallel

CMD ["python3", "benchmarks/filterpy_kaist_vio_benchmark.py", "--help"]
