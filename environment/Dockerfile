# Inference image for the CUHK-X Small Model Track solution of team Timass.
# Build context is the PACKAGE ROOT (not this environment/ directory):
#
#   docker build -f environment/Dockerfile -t timass-cuhkx .
#   docker run --rm --gpus all --network none --shm-size 16g \
#       -v <test_data_dir>:/data:ro -v <out_dir>:/out timass-cuhkx \
#       bash inference.sh /data /out/submission.csv
#   # second Selected submission: append its Submission ID as a 3rd argument
#   #   bash inference.sh /data /out/submission_w20.csv 56058031
#
# Self-contained: code, pinned dependencies and both selected packs
# (artifacts/model.pth = primary submission 56094379, artifacts/model_pack3ira.pth = submission 56058031;
# inference.sh picks the pack from its optional 3rd argument, "primary" by default).
# No network access is needed at run time. GPU: one NVIDIA GPU, >= 4 GB (peak 2.9 GB measured),
# driver with CUDA 12.6 support; without --gpus the same image runs on CPU.
FROM python:3.13.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_OFFLINE=true \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# OpenCV (imported by ultralytics) needs libGL and glib at import time
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY environment/ environment/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
        -r environment/requirements.txt -c environment/constraints-venv.txt

COPY inference.sh README.md ./
COPY src/ src/
# public repo: only the primary pack is released (see README); add model_pack3ira.pth here to run submission 56058031
COPY artifacts/model.pth artifacts/
CMD ["bash", "inference.sh"]
