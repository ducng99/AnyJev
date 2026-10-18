"""The Tacit `llamacpp` engine against a stub llama-server: prompts as token ids, labels read by id,
escalation over the native /completion endpoint -- no model, no GPU, no transformers.

The stub reproduces llama-server's shapes for a batched native request: "prompt" as an array of
token-id arrays (one task per prompt), replies that carry an "index" and collapse to a single object
when the batch has one prompt. With `n_predict=1` the reply carries the scores in
`completion_probabilities[0].top_logprobs`; with a larger `n_predict` it is a generation whose content
stops before the stop word, because llama-server does not echo it back.
"""
from __future__ import annotations

import json
import re
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from anyjev.question import Question
from anyjev.readout import DEFAULT_SYSTEM, build_prompt, render_chat, resolve_labels
from anyjev.tacit import COT_SYSTEM, LlamaCppEngine, Tacit, _render_thinking, cot_user


def logprob(tid: int, odd: float = -1.5, even: float = -0.25) -> float:
    return odd if tid % 2 else even


# the stub's whole "vocabulary": every id the tokenizer can hand over as a label, so a well-formed
# request never misses and the missing-label case is tested with an explicit override
TOP = [{"id": i, "logprob": logprob(i)} for i in range(41, 300)]
ALT = [{"id": i, "logprob": logprob(i, -0.4, -2.0)} for i in range(41, 300)]


class Tok:
    """A toy tokenizer: whitespace words, ids from 41 up (so the stub's fixed vocabulary covers the
    labels), with an exact inverse."""

    chat_template = None

    def __init__(self):
        self._vocab = {}

    def _id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = 40 + len(self._vocab) + 1
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> list:
        return [self._id(w) for w in re.findall(r" ?\S+|\s+", text)]

    def decode(self, ids) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[int(t)] for t in ids)

    def __call__(self, text: str, add_special_tokens: bool = False, **_):
        # the engines read `.input_ids` off the encode result, as on a Hugging Face tokenizer
        return types.SimpleNamespace(input_ids=self.encode(text, add_special_tokens))


class _Stub(BaseHTTPRequestHandler):
    requests: list = []                  # (body, Authorization) of every /completion POST, in order
    gen_contents: list = ["reasoning"]   # generation content per prompt, cycled
    probs = None                         # None: TOP; otherwise the top_logprobs every reply carries
    no_probs = False                     # reply without completion_probabilities at all
    per_index_alt = False                # the second prompt of a batch is scored with ALT
    reversed_multi = False               # multi-prompt replies arrive out of order

    def do_POST(self):  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.requests.append((body, self.headers.get("Authorization")))
        n = len(body["prompt"])

        def reply(i):
            generating = body.get("n_predict", 1) > 1
            out = {"index": i, "content": _Stub.gen_contents[i % len(_Stub.gen_contents)] if generating else ""}
            if not generating and not _Stub.no_probs:
                top = _Stub.probs if _Stub.probs is not None else TOP
                if _Stub.per_index_alt and i == 1:
                    top = ALT
                out["completion_probabilities"] = [{"id": 9, "top_logprobs": top}]
            return out

        out = [reply(i) for i in range(n)]
        if _Stub.reversed_multi:
            out = out[::-1]
        payload = json.dumps(out[0] if n == 1 else out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def stub():
    _Stub.requests, _Stub.gen_contents = [], ["reasoning"]
    _Stub.probs, _Stub.no_probs, _Stub.per_index_alt, _Stub.reversed_multi = None, False, False, False
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_port
    srv.shutdown()


def engine(stub, **kw):
    return LlamaCppEngine(stub, Tok(), **kw)


def tacit(stub, **kw):
    eng = engine(stub)
    kw.setdefault("max_cot_share", None)
    return Tacit(eng, eng.tok, **kw), eng


def one_forward_text(tok, state, question, options):
    q = Question.choice(question, options)
    labels, _ = resolve_labels(tok, q)
    return render_chat(tok, build_prompt(state, q, list(range(q.k)), DEFAULT_SYSTEM, labels))


def test_one_forward_posts_the_token_ids_of_the_training_prompt_and_reads_the_labels_by_id(stub):
    t, eng = tacit(stub)
    d = t.decide(state="S", question="Q?", options=["track", "refund"])
    assert d["route"] == "one_forward" and d["answer"] == "refund"
    body, auth = _Stub.requests[0]
    assert auth is None
    text = one_forward_text(eng.tok, "S", "Q?", ["track", "refund"])
    assert body == {"prompt": [eng.tok(text, add_special_tokens=False).input_ids],
                    "n_predict": 1, "temperature": 0.0, "cache_prompt": True, "n_probs": 1 << 30}
    # the labels were scored by id (-1.5 for A, -0.25 for B) and renormalized over the two of them
    w = np.exp([-1.5, -0.25])
    w = w / w.sum()
    assert d["probs"]["track"] == pytest.approx(w[0]) and d["probs"]["refund"] == pytest.approx(w[1])


def test_an_escalated_decision_generates_with_the_think_stop_and_reads_after_the_answer(stub):
    t, eng = tacit(stub, adaptive=True, tau=2.0)          # the stub's margin is 1.25, so tau=2 escalates
    d = t.decide(state="unsure", question="Q?", options=["a", "b"])
    assert d["route"] == "cot" and d["first_pass"]["route"] == "one_forward"
    assert sum(d["probs"].values()) == pytest.approx(1.0)
    q = Question.choice("Q?", ["a", "b"])
    labels, _ = resolve_labels(eng.tok, q)
    prompt = _render_thinking(eng.tok, COT_SYSTEM, cot_user("unsure", q, labels))
    gen_body, _ = _Stub.requests[1]
    assert gen_body == {"prompt": [eng.tok(prompt).input_ids], "n_predict": t.cot_max_tokens,
                        "stop": ["</think>"], "cache_prompt": True,
                        "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    # llama-server drops the stop word, so the thought is closed by hand before the read
    read_body, _ = _Stub.requests[2]
    assert eng.tok.decode(read_body["prompt"][0]) == prompt + "reasoning\n</think>\n\nAnswer:"
    assert d["reasoning_tokens"] == 1


def test_a_thought_that_carries_its_close_is_not_closed_twice(stub):
    _Stub.gen_contents = ["half a thought</think>"]       # a server that does echo the stop word
    t, eng = tacit(stub, adaptive=True, tau=2.0)
    d = t.decide(state="unsure", question="Q?", options=["a", "b"])
    assert d["route"] == "cot"
    read_body, _ = _Stub.requests[2]
    assert eng.tok.decode(read_body["prompt"][0]).endswith("half a thought</think>\n\nAnswer:")


def test_a_batch_reads_each_reply_by_its_index_not_its_position(stub):
    _Stub.per_index_alt = True
    _Stub.reversed_multi = True
    t, _ = tacit(stub)
    out = t.decide_batch([dict(state="S", question="Q?", options=["a", "b"]),
                          dict(state="T", question="Q?", options=["a", "b"])])
    # the replies arrived as (index 1, ALT), (index 0, TOP); the scores follow the index
    w0 = np.exp([-1.5, -0.25])
    w0 = w0 / w0.sum()
    w1 = np.exp([-0.4, -2.0])
    w1 = w1 / w1.sum()
    assert out[0]["probs"]["a"] == pytest.approx(w0[0]) and out[1]["probs"]["a"] == pytest.approx(w1[0])


def test_n_probs_caps_the_reply_and_a_label_outside_it_is_an_error(stub):
    _Stub.probs = TOP[:2]
    eng = engine(stub, n_probs=2)
    with pytest.raises(Exception, match=r"no log-probability for label token ids \[43\]") as info:
        eng.read(["a prompt"], [[41, 43]])
    assert type(info.value).__name__ == "LabelTokenError"
    body, _ = _Stub.requests[0]
    assert body["n_probs"] == 2


def test_a_reply_without_scores_is_an_error(stub):
    _Stub.no_probs = True
    with pytest.raises(RuntimeError, match="completion_probabilities"):
        engine(stub).read(["a prompt"], [[41]])


def test_the_api_key_is_a_bearer_token_with_an_environment_fallback(stub, monkeypatch):
    engine(stub, api_key="secret").read(["p"], [[41]])
    assert _Stub.requests[-1][1] == "Bearer secret"
    monkeypatch.setenv("TACIT_API_KEY", "env-key")
    engine(stub).read(["p"], [[41]])
    assert _Stub.requests[-1][1] == "Bearer env-key"
    monkeypatch.delenv("TACIT_API_KEY")
    engine(stub).read(["p"], [[41]])
    assert _Stub.requests[-1][1] is None


def test_from_pretrained_needs_a_base_url_and_rejects_unknown_engines():
    with pytest.raises(ValueError, match="base_url"):
        Tacit.from_pretrained("some/tacit", engine="llamacpp")
    with pytest.raises(ValueError, match="llamacpp"):
        Tacit.from_pretrained("some/tacit", engine="wat")
