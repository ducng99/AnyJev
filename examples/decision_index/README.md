# AnyJev L0 on the Decision Index

The engine behind the board entry **AnyJev L0 · Gemma-4-31B**: the `Decider`'s L0 readout of the unchanged
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), as an engine for the
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) kit. No weights are trained.
The result is in the main README ([🏆 Training-free on the Decision Index](../../README.md#-training-free-on-the-decision-index))
and in `bench/results_decision_index/2026-10-12/anyjev-l0-gemma-4-31b.json`; the model card is
[morriszjm/AnyJev-L0-Gemma-4-31B](https://huggingface.co/morriszjm/AnyJev-L0-Gemma-4-31B).

These are the files of the submitted run, unchanged. The board links to the copy it was submitted from,
[MorrisZJ/anyjev-l0-gemma@50476e9](https://github.com/MorrisZJ/anyjev-l0-gemma/tree/50476e9), which stays as it
is; this folder is where the engine is maintained and where a later edition is run from.

| file | |
|---|---|
| `anyjev_engine.py` | the engine, `anyjev_engine:AnyJevEngine` |
| `serve.sh` | the vLLM server it reads: FP8 at load, `--logprobs-mode processed_logprobs` |
| `run.sh` | the public 0.3 suite through the engine, then the kit's scorer |
| `requirements.txt` | anyjev 0.3.0, vLLM 0.29.0, the kit at `9eb2dbe` |

## Run it

```bash
cd examples/decision_index
pip install -r requirements.txt
git clone https://github.com/apolinario/decision-index && git -C decision-index checkout 9eb2dbe
pip install -e decision-index        # then build the suite as the kit's README describes
bash serve.sh                        # one GPU; TP=2 MAXLEN=131072 is the setting of the submitted run
bash run.sh
```

## How it reads a decision

- Each question is rendered with AnyJev's prompt and read as the next-token distribution restricted to its option
  labels, one read per rotation. Nothing is generated; a prompt longer than the server's window is refused, not cut.
- L0: two cyclic rotations averaged in log space, and the label prior `Decider` keeps for a question once it has
  seen it 8 times. The prior is kept per process, so a question's distribution depends slightly on the requests that
  came before it.
- Above 26 options (the kit allows 255) each option gets a letter and a digit, and
  p(option) = p(letter) · p(digit | letter). The library's `Decider` stops at 26 options; this readout is on the
  roadmap as the span readout.
- Yes/no questions are shown with their criteria ("Yes means: … No means: …").

The engine imports the kit (`decision_index`), so it is an example, not part of the `anyjev` package, and CI does not
run it. The kit is MIT-licensed ([THIRD_PARTY.md](../../THIRD_PARTY.md)).
