# LEARN — Triton serving, the concepts behind this lab

Read this alongside running the lab. By the end you should be able to answer the self-quiz cold — those are
close to what NVIDIA will actually ask.

## 1. Why serve through Triton at all?
Training gives you a `model.pt`. Production needs a *service*: something that accepts many concurrent requests
over the network, runs them efficiently on the hardware, and stays up. Triton is that service layer. It supports
multiple backends (ONNX, TensorRT, PyTorch, TensorFlow, Python) behind one HTTP/gRPC surface, and adds the
production features you'd otherwise hand-roll: batching, multiple model instances, versioning, metrics.

**The request path (memorize this):**
`client → HTTP:8000 / gRPC:8001 → per-model scheduler → dynamic batcher → backend (ONNX Runtime) → response`.
Port 8002 serves Prometheus metrics.

## 2. Why ONNX in the middle?
ONNX is a portable graph format. `torch.onnx.export` traces your PyTorch model into a framework-agnostic graph
that the ONNX-Runtime backend can execute without PyTorch installed. It's also the on-ramp to **TensorRT**
(NVIDIA's optimizing compiler) — ONNX → TensorRT is the standard high-performance path. So ONNX is both "runs
anywhere" and "the doorway to the fast NVIDIA path."

**The one gotcha:** the batch axis must be *dynamic*. We pass `dynamic_axes={"input": {0: "batch"}}` so axis 0 is
symbolic. If it were fixed at 1, Triton could never batch — every request would be locked to batch size 1.

## 3. Dynamic batching — the core idea of this lab
Under load, requests arrive one at a time. Two ways to handle them:
- **No batching:** run a separate forward pass per request. Simple, but each pass under-uses the hardware.
- **Dynamic batching:** hold arriving requests for up to `max_queue_delay_microseconds`, fuse whatever showed up
  into a single batch (aiming for `preferred_batch_size`), run one forward pass, then split the results back out.

Why it wins: a forward pass has fixed overhead (kernel launches, memory setup). Amortizing that over 8 samples
instead of 1 raises throughput, and once you're under queueing load it often *lowers* average latency too. The
trade is a few ms of deliberate waiting. That `max_queue_delay` is the latency-vs-throughput dial.

This is why the lab ships two identical models differing only in the `dynamic_batching` block — so you can
*measure* the difference, not just believe it.

## 4. Compute-bound vs memory-bound (bridge to Dynamo / LLMs)
This CNN is **compute-bound** — the bottleneck is matrix-multiply FLOPs, which is exactly why batching helps
(more math per fixed overhead). LLM decoding is different: it's **memory-bound** — each new token must re-read the
growing **KV cache** from memory, so the bottleneck is memory bandwidth, not FLOPs. That single distinction is why
Dynamo splits **prefill** (compute-bound) from **decode** (memory-bound) onto separately scaled GPU pools. You just
built the compute-bound half's intuition; keep that thread for Deliverable 2.

## 5. Self-quiz (answer out loud before interviews)
1. Trace a request from `client.py` to a returned logit vector. Name every stage.
2. Why must the ONNX batch axis be dynamic? What breaks if it's fixed at 1?
3. `max_queue_delay_microseconds` is the knob for what trade-off? Which way do you turn it for a strict-latency SLO?
4. Your CNN benchmark shows batching raises throughput. Would you expect the *same* win for single-stream LLM
   decode? Why or why not? (hint: compute- vs memory-bound)
5. What do ports 8000 / 8001 / 8002 do? When would you pick gRPC over HTTP?
6. Where does TensorRT fit relative to ONNX in the NVIDIA serving path?

## 6. What this unlocks on your resume (only after you run it and fill the table)
> Served an ONNX-exported CNN via NVIDIA Triton Inference Server with dynamic batching; benchmarked
> latency (p50/p95/p99) and throughput with vs without batching, quantifying the throughput gain from
> request fusion.

## 7. Next
- Swap in your real trained weights: `python export_onnx.py --weights your_model.pt` (architecture must match).
- Optional depth: convert the ONNX model to **TensorRT** and re-benchmark — that's the "GPU memory management /
  high-performance" bullet the JD lists under "ways to stand out."
- Then Deliverable 2 (run Dynamo, write up disaggregated serving) — see the Playbook §2.
