"""llama.cpp backend over llama-server's native `/completion` endpoint.

One request per prompt: `n_predict=1` and `n_probs` large. The token the server samples for
that one step is discarded; what is read is `completion_probabilities[0].top_logprobs`, which
llama-server fills with the plain softmax over the raw logits (`post_sampling_probs` stays
false, and any positive `n_probs` routes the request around the backend sampler, so no
sampler setting can bend the numbers). The default asks for the whole vocabulary, so the
values are the full-vocabulary log-softmax of the requested ids -- the same numbers
`HFBackend` reports -- and a label can never be missing. Pass `n_probs` (a few times the
label count) to shrink the reply to the top N tokens; a label outside the top N then raises
`LabelTokenError` rather than being guessed.

    llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --alias qwen2.5-7b
    Decider(LlamaCppServerBackend("http://127.0.0.1:8080", "qwen2.5-7b"))

Tokenizer. By default the server's own vocabulary is used, through its `/tokenize`,
`/detokenize` and `/apply-template` endpoints: no transformers install and no local GGUF are
needed, the label ids come from the same vocabulary the server tokenizes the prompt with,
and prompts rendered by readout go through the server's chat template via `/apply-template`,
which always appends the generation prompt and takes no template variables (`enable_thinking`
is not forwarded). `model` is required and names the served model on every request, as a
llama-server router needs (the server resolves aliases itself). With `tokenizer_name` (a
Hugging Face name) that tokenizer renders the prompts and maps the labels, exactly as
`VLLMBackend` does -- use it when the offset mapping of a fast tokenizer is wanted for the
feature caches.

Unlike vLLM's `--logprobs-mode processed_logprobs` these are full-vocabulary log-probabilities,
not renormalized over the labels. `cache_prompt` (the server default, kept on) reuses the KV of
shared prefixes, so the K permutations of one state cost one prefill plus cheap suffixes. No
hidden states: an L2 reader needs `HFBackend` or a vLLM embed server. Engine smoke test:
`tests/test_llamacpp_server.py`; parity against transformers: `scripts/llamacpp_server_parity.py`.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from anyjev.readout import LabelTokenError

# more than any vocabulary: llama-server clamps n_probs to the vocabulary size, so this asks
# for the whole distribution without having to know n_vocab
_ALL_TOKENS = 1 << 30


def _post_json(url: str, body: Dict[str, Any], api_key: Optional[str], timeout: float):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from e


class LlamaCppServerTokenizer:
    """The server's own tokenizer, over its `/tokenize`, `/detokenize` and `/apply-template`
    endpoints. `chat_template` is a non-empty marker: the template itself stays on the server
    and is applied by `/apply-template`, so readout routes rendering there instead of falling
    back to plain text."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 timeout: float = 120.0, chat_template: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.chat_template = chat_template or "server-side"
        self.model = model

    def _body(self, **fields: Any) -> Dict[str, Any]:
        fields["model"] = self.model
        return fields

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # add_special False: no BOS, as an HF encode with add_special_tokens=False; the server
        # decides on the BOS itself when it tokenizes a string prompt for /completion
        out = _post_json(self.base_url + "/tokenize",
                         self._body(content=text, add_special=bool(add_special_tokens)),
                         self.api_key, self.timeout)
        return [int(t) for t in out["tokens"]]

    def decode(self, ids: Sequence[int]) -> str:
        out = _post_json(self.base_url + "/detokenize",
                         self._body(tokens=[int(t) for t in ids]),
                         self.api_key, self.timeout)
        return out["content"]

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False,
                            add_generation_prompt: bool = True, **kwargs: Any):
        """The server renders with its own template and appends the generation prompt (the
        endpoint exists to prefill the assistant turn); `add_generation_prompt=False` cannot be
        honored and raises rather than rendering something else."""
        if not add_generation_prompt:
            raise ValueError("/apply-template always appends the generation prompt; "
                             "add_generation_prompt=False is not supported by this tokenizer")
        out = _post_json(self.base_url + "/apply-template",
                         self._body(messages=list(messages)),
                         self.api_key, self.timeout)
        text = out["prompt"]
        return self.encode(text) if tokenize else text


class LlamaCppServerBackend:
    """Next-token log-probabilities from a running llama-server.

    `model` is the served model name, required because a llama-server router rejects requests
    that do not name one. `api_key` is sent as a bearer token when the server was started with
    `--api-key`. `n_probs=None` scores the whole vocabulary (exact, larger replies); an integer
    caps the reply at the top N tokens.
    """

    def __init__(self, base_url: str, model: str, tokenizer_name: str | None = None,
                 api_key: str | None = None, workers: int = 16,
                 timeout: float = 120.0, n_probs: int | None = None):
        if not model:
            raise ValueError("model is required: name the served model (a llama-server router rejects "
                             "requests without one)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        self.n_probs = n_probs
        if tokenizer_name is None:
            self.tokenizer = LlamaCppServerTokenizer(self.base_url, model, api_key, timeout)
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _one(self, prompt: str, ids: Sequence[int]) -> np.ndarray:
        body = {
            "model": self.model,
            "prompt": prompt, "n_predict": 1, "temperature": 0.0, "cache_prompt": True,
            "n_probs": _ALL_TOKENS if self.n_probs is None else int(self.n_probs),
        }
        out = _post_json(self.base_url + "/completion", body, self.api_key, self.timeout)
        try:
            top = out["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"{self.model}: the reply carries no completion_probabilities for the first predicted token; "
                "n_probs must be positive on a /completion request of a recent llama-server") from e
        by_id = {}
        for entry in top:
            if entry.get("logprob") is not None:
                by_id[int(entry["id"])] = float(entry["logprob"])
        missing = [int(t) for t in ids if int(t) not in by_id]
        if missing:
            raise LabelTokenError(
                f"{self.model}: no log-probability for label token ids {missing}; the server returned "
                f"its top {len(top)} tokens (n_probs={self.n_probs}). Leave n_probs unset to score the "
                f"whole vocabulary, or raise n_probs")
        return np.array([by_id[int(t)] for t in ids], dtype=np.float64)

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        with cf.ThreadPoolExecutor(self.workers) as executor:
            return list(executor.map(self._one, prompts, token_ids))
