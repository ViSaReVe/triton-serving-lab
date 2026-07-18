#!/usr/bin/env bash
# Start Triton locally with the two model configs mounted.
#
# Apple-Silicon Mac note: the official Triton image is x86_64, so Docker runs it under emulation
# (--platform linux/amd64). It WORKS but inference is slow — that's fine for learning the pipeline
# and seeing the batching-vs-nobatching *shape*. For clean absolute numbers, run the same thing on
# an x86 box or Colab (see README "Clean numbers on Colab").
#
# Tag: any recent "*-py3" works. 24.12-py3 is a safe known-good; bump to a newer tag if you like.
set -euo pipefail

TAG="${TRITON_TAG:-24.12-py3}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)/model_repository"

echo "Serving models from: $REPO_DIR"
echo "Image: nvcr.io/nvidia/tritonserver:$TAG  (first pull is several GB)"

docker run --rm -it \
  --platform linux/amd64 \
  --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$REPO_DIR:/models" \
  nvcr.io/nvidia/tritonserver:"$TAG" \
  tritonserver --model-repository=/models --model-control-mode=none

# Ports:  8000 = HTTP/REST   8001 = gRPC   8002 = Prometheus metrics
# When you see "started GRPCInferenceService" and both models READY, open another terminal and:
#   python client.py
#   python benchmark.py --model audio_cnn          --concurrency 16 --requests 500
#   python benchmark.py --model audio_cnn_nobatch  --concurrency 16 --requests 500
