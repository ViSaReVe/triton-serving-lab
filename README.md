# triton-serving-lab

Serve a PyTorch model with **NVIDIA Triton Inference Server** via the **ONNX Runtime** backend, turn on
**dynamic batching**, and measure the latency/throughput trade-off. Built to understand the Triton serving
path end-to-end (the exact stack behind NVIDIA's Dynamo-Triton team).

```
PyTorch model ──torch.onnx.export──▶ model.onnx (dynamic batch axis)
                                          │
                          Triton model_repository/<name>/{config.pbtxt, 1/model.onnx}
                                          │
client ──HTTP:8000 / gRPC:8001──▶ Triton ──▶ per-model scheduler ──▶ dynamic batcher ──▶ ONNX-Runtime backend ──▶ logits
                                                                         (fuses many requests into one forward pass)
```

## Results (fill these in after you run it)

CPU, 500 requests/config. *(Apple-Silicon numbers are emulated — see note below; use Colab for clean numbers.)*

| model | concurrency | throughput (req/s) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| audio_cnn (dynamic batching) | 16 | _ | _ | _ | _ |
| audio_cnn_nobatch (control) | 16 | _ | _ | _ | _ |

**Expected finding:** as concurrency rises, `audio_cnn` (batching on) delivers higher throughput because Triton
fuses concurrent single-sample requests into one forward pass — fewer, larger passes beat many tiny ones.

## Quickstart
```bash
pip install -r requirements.txt

# 1) export the model to ONNX (writes into both model_repository configs) + local sanity check
python export_onnx.py

# 2) start Triton (downloads a few GB the first time)
bash run_local.sh            # Ctrl-C to stop

# 3) in a second terminal — prove it answers, then benchmark both configs
python client.py
python benchmark.py --model audio_cnn         --concurrency 16 --requests 500
python benchmark.py --model audio_cnn_nobatch --concurrency 16 --requests 500
```

## Apple-Silicon Mac note (honest caveat, keep it in the README)
The official Triton image is x86_64, so on an M-series Mac Docker runs it under `--platform linux/amd64`
emulation. The pipeline works and the batching-vs-no-batching *shape* is visible, but absolute latency is
emulation-inflated. This is a genuine, disclosable limitation — the kind of honesty that reads well.

### Clean numbers on Colab
Open a free Colab notebook and run Triton natively on x86 (optionally a T4 GPU):
`docker` isn't available in Colab, so use the `tritonserver` pip path or run `perf_analyzer` from the
Triton SDK — or simplest: run this repo on any x86 cloud box. Re-run `benchmark.py` there for the table.

## Files
- `model.py` — SimpleAudioCNN (swap in your UrbanSound/EMG weights later via `export_onnx.py --weights`)
- `export_onnx.py` — PyTorch → ONNX with a dynamic batch axis; verifies load in onnxruntime
- `model_repository/audio_cnn/` — config **with** `dynamic_batching`
- `model_repository/audio_cnn_nobatch/` — identical model, **no** batching (the control)
- `client.py` — single-request smoke test
- `benchmark.py` — concurrent load test → p50/p95/p99 + throughput (`--selftest` runs offline)
- `run_local.sh` — the `docker run` command with the right ports/mounts

See `LEARN.md` for the concept walkthrough and self-quiz.
