"""
Minimal Triton client — send ONE request and print the result.
Use this first to confirm the server is up and the model answers.

    pip install tritonclient[http] numpy
    python client.py                       # defaults to model audio_cnn on localhost:8000
    python client.py --model audio_cnn_nobatch
"""
import argparse

import numpy as np
import tritonclient.http as httpclient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model", default="audio_cnn")
    args = ap.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)

    # one fake log-mel spectrogram: (batch=1, 1 channel, 128 mel bins, 128 frames)
    x = np.random.randn(1, 1, 128, 128).astype(np.float32)

    inp = httpclient.InferInput("input", x.shape, "FP32")
    inp.set_data_from_numpy(x)
    out = httpclient.InferRequestedOutput("logits")

    resp = client.infer(model_name=args.model, inputs=[inp], outputs=[out])
    logits = resp.as_numpy("logits")
    print("logits shape:", logits.shape)          # (1, 10)
    print("predicted class:", int(logits.argmax()))
    print("server says the model is ready:", client.is_model_ready(args.model))


if __name__ == "__main__":
    main()
