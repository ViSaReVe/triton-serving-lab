"""
Load-test a Triton model and report p50 / p95 / p99 latency + throughput.

Why this instead of perf_analyzer? perf_analyzer lives in a separate multi-GB Triton SDK
container. This pure-Python version does the same core job (fire many concurrent single-sample
requests, time each one) and runs anywhere you have tritonclient. The NUMBERS are what go in
your README table.

    pip install tritonclient[http] numpy
    # server must be running first (see run_local.sh)
    python benchmark.py --model audio_cnn           --concurrency 16 --requests 500
    python benchmark.py --model audio_cnn_nobatch   --concurrency 16 --requests 500
Then compare the two: dynamic batching should win on throughput as concurrency rises.

--selftest runs the statistics math with a fake client (no server needed) so you can verify
the harness logic on its own.
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def percentile(values, p):
    """Nearest-rank percentile (no numpy dependency in the hot path)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def run_load(infer_fn, n_requests, concurrency):
    """infer_fn() performs one blocking inference and returns None. Returns per-request latencies (ms)."""
    latencies = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_timed, infer_fn) for _ in range(n_requests)]
        for f in as_completed(futures):
            latencies.append(f.result())
    wall = time.perf_counter() - t_start
    return latencies, wall


def _timed(infer_fn):
    t0 = time.perf_counter()
    infer_fn()
    return (time.perf_counter() - t0) * 1000.0  # ms


def report(name, latencies, wall, concurrency):
    thr = len(latencies) / wall if wall > 0 else float("nan")
    print(f"\n=== {name} | concurrency={concurrency} | n={len(latencies)} ===")
    print(f"  throughput : {thr:8.1f} req/s   (wall {wall:.2f}s)")
    print(f"  latency p50: {percentile(latencies,50):8.2f} ms")
    print(f"  latency p95: {percentile(latencies,95):8.2f} ms")
    print(f"  latency p99: {percentile(latencies,99):8.2f} ms")
    print(f"  latency avg: {statistics.mean(latencies):8.2f} ms   max {max(latencies):.2f} ms")
    return {"model": name, "concurrency": concurrency, "throughput_rps": round(thr, 1),
            "p50_ms": round(percentile(latencies, 50), 2), "p95_ms": round(percentile(latencies, 95), 2),
            "p99_ms": round(percentile(latencies, 99), 2)}


def make_triton_infer(url, model):
    import tritonclient.http as httpclient
    client = httpclient.InferenceServerClient(url=url, concurrency=64)
    x = np.random.randn(1, 1, 128, 128).astype(np.float32)

    def infer():
        inp = httpclient.InferInput("input", x.shape, "FP32")
        inp.set_data_from_numpy(x)
        out = httpclient.InferRequestedOutput("logits")
        client.infer(model_name=model, inputs=[inp], outputs=[out])

    return infer


def selftest():
    """No server needed — verify the harness + percentile math with a fake latency profile."""
    import random

    def fake_infer():
        time.sleep(random.uniform(0.001, 0.004))  # 1-4 ms fake work

    lat, wall = run_load(fake_infer, n_requests=200, concurrency=8)
    row = report("SELFTEST(fake)", lat, wall, 8)
    assert len(lat) == 200
    assert row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"], "percentiles must be monotonic"
    assert row["throughput_rps"] > 0
    # sanity on percentile() with a known list
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) in (5, 6)
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100) == 10
    print("\nSELFTEST PASSED: harness + percentile math OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", default="audio_cnn")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    infer = make_triton_infer(args.url, args.model)
    infer()  # warm-up (loads model, first-call overhead)
    lat, wall = run_load(infer, args.requests, args.concurrency)
    report(args.model, lat, wall, args.concurrency)


if __name__ == "__main__":
    main()
