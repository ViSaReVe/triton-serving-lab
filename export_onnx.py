"""
Export the EE541 SimpleCNN -> ONNX, place it in BOTH Triton model repositories, and verify it loads.

Why a dynamic batch axis?  Triton's dynamic batcher combines many single-sample requests into one
batch on the fly. For that to work, the ONNX graph must accept a variable batch dimension — that's the
`dynamic_axes={... 0: "batch"}` below (axis 0 = N is left symbolic).

SimpleCNN.forward() returns ONLY logits, so no output wrapper is needed.

Run:
    python export_onnx.py                                   # random weights (fine for a serving benchmark)
    python export_onnx.py --weights E3_mixup_fold1_best.pt  # your trained checkpoint (real logits)
"""
import argparse
import pathlib
import shutil

import torch

from model import SimpleCNN

REPO = pathlib.Path(__file__).parent / "model_repository"
TARGETS = ["audio_cnn", "audio_cnn_nobatch"]  # batching ON and OFF — same weights, two configs


def load_checkpoint(model, path):
    """Handle both {'model_state_dict': ...} and raw state_dict checkpoints."""
    state = torch.load(path, map_location="cpu")
    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {path} | missing={list(missing)} unexpected={list(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="optional .pt checkpoint (E3_mixup_fold*_best.pt)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = SimpleCNN(num_classes=10).eval()
    if args.weights:
        load_checkpoint(model, args.weights)
    else:
        print("using random weights (serving benchmark does not need trained weights)")

    dummy = torch.randn(1, 1, 128, 128)  # (N, C, mel_bins, frames)

    tmp = REPO / "_model.onnx"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, tmp.as_posix(),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"exported ONNX (opset {args.opset})")

    for name in TARGETS:
        dst = REPO / name / "1"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp, dst / "model.onnx")
        print(f"  -> {dst / 'model.onnx'}")
    tmp.unlink()

    # verify it loads + honors the dynamic batch axis (fast local check, no Docker needed)
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession((REPO / "audio_cnn" / "1" / "model.onnx").as_posix(),
                                    providers=["CPUExecutionProvider"])
        for n in (1, 4):
            out = sess.run(None, {"input": np.random.randn(n, 1, 128, 128).astype("float32")})[0]
            assert out.shape == (n, 10), out.shape
            print(f"  onnxruntime OK: input (N={n}) -> logits {out.shape}")
        print("VERIFIED: ONNX model loads and honors the dynamic batch axis.")
    except ImportError:
        print("(install onnxruntime to run the local sanity check: pip install onnxruntime)")


if __name__ == "__main__":
    main()
