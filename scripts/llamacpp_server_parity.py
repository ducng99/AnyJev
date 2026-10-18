"""Backend parity: LlamaCppServerBackend (llama-server) vs HFBackend on the same prompts.

    llama-server -m qwen2.5-0.5b-instruct-f16.gguf --alias qwen2.5-0.5b --port 8080
    python scripts/llamacpp_server_parity.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --served-model qwen2.5-0.5b --base-url http://127.0.0.1:8080 \
        --device cpu --dtype float32 --json parity.json

Both sides get the prompts rendered by the Hugging Face tokenizer and the same label ids
(the server backend is built with `tokenizer_name=model`); the server answers with the plain
softmax over its raw logits, so nothing is renormalized on either side. Reports max |diff| of
the full-vocabulary log-probs of the label tokens, of the log-softmax restricted to the
labels, and argmax agreement over choice / noul / score prompts. For parity the GGUF must be
a faithful conversion of the same Hugging Face snapshot the HF side loads: a quantized file
also measures llama.cpp's reduced-precision compute, exactly as for the in-process backend
(scripts/llamacpp_parity.py has the numbers).
"""
from __future__ import annotations

import argparse
import json
import platform

import numpy as np

from anyjev import Question
from anyjev.readout import build_prompt, label_ids_for_perm, render_chat, resolve_labels
from anyjev.state import render_state


def lsm(x):
    x = np.asarray(x, float)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="the HF model the served GGUF was converted from")
    ap.add_argument("--served-model", default=None,
                    help="the model name llama-server serves (its --alias); defaults to --model")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json", default=None, help="also write the numbers and the environment here")
    args = ap.parse_args()
    import torch
    import transformers

    from anyjev.backends.hf import HFBackend
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    hf = HFBackend(args.model, device=args.device, dtype=args.dtype, batch_size=8)
    srv = LlamaCppServerBackend(args.base_url, args.served_model or args.model, tokenizer_name=args.model)
    qs = [Question.choice("Which team handles this?", ["billing", "technical", "sales", "other"]),
          Question.noul("Is this message a complaint?"),
          Question.score("How urgent is this?", levels=["not", "low", "medium", "high"])]
    states = ["My card was charged twice.", "The app crashes on login.", "Do you offer bulk discounts?",
              "Thanks, all sorted now!", "URGENT: production is down for all users."]
    prompts, ids = [], []
    for q in qs:
        _, label_ids = resolve_labels(hf.tokenizer, q)
        perm = list(range(q.k))
        for state in states:
            prompts.append(render_chat(hf.tokenizer, build_prompt(render_state(state), q, perm)))
            ids.append(label_ids_for_perm(q, label_ids, perm))
    expected = hf.next_token_logprobs(prompts, ids)
    actual = srv.next_token_logprobs(prompts, ids)
    diffs, restricted, agree = [], [], 0
    for left, right in zip(expected, actual):
        diffs.append(float(np.max(np.abs(left - right))))
        restricted.append(float(np.max(np.abs(lsm(left) - lsm(right)))))
        agree += int(np.argmax(left) == np.argmax(right))
    print(f"prompts={len(prompts)} argmax_agreement={agree}/{len(prompts)} "
          f"max_abs_diff={max(diffs):.4f} mean={np.mean(diffs):.4f} "
          f"max_restricted_diff={max(restricted):.4f}")
    print("server sample:", np.round(actual[0], 3), "hf:", np.round(expected[0], 3))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"prompts": len(prompts), "argmax_agreement": agree,
                       "max_abs_diff": max(diffs), "mean_abs_diff": float(np.mean(diffs)),
                       "max_restricted_diff": max(restricted),
                       "model": args.model, "base_url": args.base_url,
                       "device": args.device, "dtype": args.dtype,
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "python": platform.python_version()}, f, indent=2)


if __name__ == "__main__":
    main()
