"""llama-server API contract tests without a llama.cpp install, a GPU or transformers.

The stub answers with the response shape llama-server returns for a ``/completion`` call with
``n_predict=1`` and ``n_probs > 0``: the scores sit under ``completion_probabilities[0]`` as
``top_logprobs`` entries of ``{id, token, bytes, logprob}``, where ``logprob`` is the plain
softmax of the raw logits (``post_sampling_probs`` false). The backend itself only speaks
HTTP, so the real class is constructed against the stub -- nothing is downloaded.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

REAL_SHAPE = {
    "completion_probabilities": [
        {
            "id": 42, "token": " B", "bytes": [32, 66], "logprob": -0.25,
            # deliberately not in the order the labels were requested: they are read by id
            "top_logprobs": [
                {"id": 42, "token": " B", "bytes": [32, 66], "logprob": -0.25},
                {"id": 41, "token": " A", "bytes": [32, 65], "logprob": -1.5},
                {"id": 44, "token": " C", "bytes": [32, 67], "logprob": -9.0},
            ],
        }
    ]
}

MODEL = "qwen2.5-0.5b-instruct-f16"

RESPONSES = {
    "/completion": REAL_SHAPE,
    "/tokenize": {"tokens": [7, 8]},
    "/detokenize": {"content": "hello"},
    "/apply-template": {"prompt": "rendered prompt"},
}


class _Stub(BaseHTTPRequestHandler):
    request_body = None
    request_path = None
    auth_header = None

    def _reply(self, path):
        _Stub.request_path = path
        _Stub.auth_header = self.headers.get("Authorization")
        payload = json.dumps(RESPONSES[path]).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - http.server's name
        self._reply(self.path)

    def do_POST(self):  # noqa: N802 - http.server's name
        _Stub.request_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._reply(self.path)

    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture()
def stub():
    _Stub.request_body, _Stub.request_path, _Stub.auth_header = None, None, None
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_scores_the_labels_from_the_full_vocabulary_and_maps_the_reply_back_by_id(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    got = LlamaCppServerBackend(stub, MODEL).next_token_logprobs(["prompt"], [[41, 42, 44]])

    # one step whose sampled token is discarded, and the whole vocabulary asked for, so no
    # label can be missing and the values are the raw-logits log-softmax
    assert _Stub.request_body == {
        "model": "qwen2.5-0.5b-instruct-f16",
        "prompt": "prompt", "n_predict": 1, "temperature": 0.0, "cache_prompt": True,
        "n_probs": 1 << 30,
    }
    # the server listed B before A and added C; the result follows the requested order
    np.testing.assert_allclose(got, [[-1.5, -0.25, -9.0]])
    assert got[0].dtype == np.float64


def test_n_probs_is_passed_through_and_a_label_outside_the_top_n_is_an_error(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    short = {"id": 42, "top_logprobs": REAL_SHAPE["completion_probabilities"][0]["top_logprobs"][:2]}
    RESPONSES["/completion"] = {"completion_probabilities": [short]}
    try:
        with pytest.raises(Exception, match="no log-probability for label token ids \\[44\\]") as info:
            LlamaCppServerBackend(stub, MODEL, n_probs=2).next_token_logprobs(["prompt"], [[41, 42, 44]])
        assert type(info.value).__name__ == "LabelTokenError"
        assert _Stub.request_body["n_probs"] == 2
    finally:
        RESPONSES["/completion"] = REAL_SHAPE


def test_a_reply_without_completion_probabilities_is_an_error(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    for broken in ({}, {"completion_probabilities": []}):
        RESPONSES["/completion"] = broken
        try:
            with pytest.raises(RuntimeError, match="completion_probabilities"):
                LlamaCppServerBackend(stub, MODEL).next_token_logprobs(["prompt"], [[41]])
        finally:
            RESPONSES["/completion"] = REAL_SHAPE


def test_a_null_logprob_counts_as_missing(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    RESPONSES["/completion"] = {"completion_probabilities": [
        {"id": 41, "top_logprobs": [{"id": 41, "token": " A", "bytes": [32, 65], "logprob": None}]}]}
    try:
        with pytest.raises(Exception, match="no log-probability for label token ids \\[41\\]"):
            LlamaCppServerBackend(stub, MODEL).next_token_logprobs(["prompt"], [[41]])
    finally:
        RESPONSES["/completion"] = REAL_SHAPE


def test_batches_keep_their_order(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    got = LlamaCppServerBackend(stub, MODEL, workers=2).next_token_logprobs(["a", "b"], [[41], [42, 41]])
    np.testing.assert_allclose(got[0], [-1.5])
    np.testing.assert_allclose(got[1], [-0.25, -1.5])


def test_the_api_key_is_sent_as_a_bearer_token_and_omitted_without_one(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend(stub, MODEL, api_key="secret")
    backend.next_token_logprobs(["p"], [[41]])
    assert _Stub.auth_header == "Bearer secret"
    LlamaCppServerBackend(stub, MODEL).next_token_logprobs(["p"], [[41]])
    assert _Stub.auth_header is None


def test_the_tokenizer_is_the_server_vocabulary_over_its_own_endpoints(stub):
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    be = LlamaCppServerBackend(stub, MODEL)
    assert isinstance(be.tokenizer.chat_template, str)
    assert be.tokenizer.encode("prompt") == [7, 8]
    assert _Stub.request_body == {"content": "prompt", "add_special": False,
                                  "model": "qwen2.5-0.5b-instruct-f16"}
    assert be.tokenizer.encode("prompt", add_special_tokens=True) == [7, 8]
    assert _Stub.request_body == {"content": "prompt", "add_special": True,
                                  "model": "qwen2.5-0.5b-instruct-f16"}
    assert be.tokenizer.decode([7, 8]) == "hello"
    assert be.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True) == "rendered prompt"
    assert _Stub.request_body == {"messages": [{"role": "user", "content": "hi"}],
                                  "model": "qwen2.5-0.5b-instruct-f16"}
    assert be.tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True) == [7, 8]
    with pytest.raises(ValueError, match="generation prompt"):
        be.tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=False)


def test_no_transformers_is_needed_without_tokenizer_name(stub, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)   # any import of it raises
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    be = LlamaCppServerBackend(stub, MODEL)
    np.testing.assert_allclose(be.next_token_logprobs(["p"], [[41]])[0], [-1.5])


def test_tokenizer_name_loads_a_huggingface_tokenizer(stub, monkeypatch):
    from anyjev.backends.fake import FakeTokenizer

    class _Auto:
        @staticmethod
        def from_pretrained(name, *args, **kwargs):
            assert name == "some/real-model"
            return FakeTokenizer()

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = _Auto
    monkeypatch.setitem(sys.modules, "transformers", fake)

    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    be = LlamaCppServerBackend(stub, MODEL, tokenizer_name="some/real-model")
    assert be.tokenizer.__class__ is FakeTokenizer


def test_the_model_name_is_required():
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    with pytest.raises(ValueError, match="model is required"):
        LlamaCppServerBackend("http://127.0.0.1:1", "")


@pytest.mark.engine
def test_real_llama_server_returns_requested_logprobs():
    base_url = os.environ.get("LLAMACPP_SERVER_BASE_URL")
    model = os.environ.get("LLAMACPP_SERVER_MODEL")
    if not base_url or not model:
        pytest.skip("set LLAMACPP_SERVER_BASE_URL and LLAMACPP_SERVER_MODEL to run the llama-server "
                    "engine smoke test")
    from anyjev.backends.llamacpp_server import LlamaCppServerBackend

    backend = LlamaCppServerBackend(base_url, model,
                                    tokenizer_name=os.environ.get("LLAMACPP_SERVER_TOKENIZER"))
    ids = [backend.tokenizer.encode(" A", add_special_tokens=False)[-1],
           backend.tokenizer.encode(" B", add_special_tokens=False)[-1]]
    got = backend.next_token_logprobs(["Answer with one letter:"], [ids])[0]
    assert got.shape == (2,) and np.all(np.isfinite(got))
