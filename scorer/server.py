# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["mlx-lm>=0.28"]
# ///
"""Local token-probability scorer for aidetector.

Loads a "performer" language model and, optionally, an "observer" from the
same family sharing its tokenizer (Binoculars uses a base model as observer and
its instruction-tuned twin as performer). For each text it computes zero-shot
statistics from the full next-token distributions:

  log_ppl     performer's mean negative log-likelihood of the text's tokens
  x_ppl       cross-entropy between the observer's predictions and the
              performer's log-probabilities (observer = performer if none)
  binoculars  log_ppl / x_ppl (Hans et al., 2024). Low means the text is more
              predictable than the models expect text to be (AI-like).
  curvature   Fast-DetectGPT's analytic conditional probability curvature
              (Bao et al., 2024), with the observer as reference distribution
              and the performer as scorer. High means AI-like.

With a single chat-tuned model these flip direction between generator
families; the two-model form is what makes Binoculars family-independent.

Results are cached on disk by model pair and text hash, since fitting rescans
the corpus many times.

    uv run scorer/server.py --observer ~/.cache/aidetector/models/Qwen3-8B-Base-8bit --performer mlx-community/Qwen3-8B-8bit

(8-bit: about 17 GB for the pair. The bf16 pair is about 30 GB plus cache,
which crowded other apps out of memory on a 128 GB Mac. The 8-bit base
model is made locally with mlx_lm.convert -q --q-bits 8.)

Then set AIDETECTOR_SCORER_URL=http://127.0.0.1:8765 for the Go tools.
"""

import argparse
import hashlib
import json
import math
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_lm import load

MAX_TOKENS = 512  # score the first window, a common detector window size
MIN_TOKENS = 16


class Scorer:
    def __init__(self, performer: str, observer: str | None, cache_path: str, cache_limit_gb: float = 2.0):
        # Cap MLX's GPU buffer cache: it grows with every new input shape and
        # is pinned memory macOS can't swap, which starved other apps
        # (2026-11-02). The models themselves are the bulk of the footprint.
        mx.set_cache_limit(int(cache_limit_gb * (1 << 30)))
        self.model, self.tokenizer = load(performer)
        self.observer = load(observer)[0] if observer else None
        name = lambda p: os.path.basename(os.path.normpath(p))
        self.model_id = name(performer) if not observer else f"{name(observer)}+{name(performer)}"
        self.lock = threading.Lock()  # one forward pass at a time on the GPU
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT)")

    def score(self, text: str) -> dict:
        key = hashlib.sha256(f"v2\0{self.model_id}\0{MAX_TOKENS}\0{text}".encode()).hexdigest()
        row = self.db.execute("SELECT value FROM scores WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        result = self._compute(text)
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?)", (key, json.dumps(result)))
            self.db.commit()
        return result

    def _compute(self, text: str) -> dict:
        ids = self.tokenizer.encode(text)
        # Start from a document boundary so the first token has context:
        # the BOS token if the model has one, else Qwen-style <|endoftext|>.
        start = self.tokenizer.bos_token_id
        if start is None:
            start = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(start, int) and start >= 0 and (not ids or ids[0] != start):
            ids = [start] + ids
        ids = ids[: MAX_TOKENS + 1]
        n = len(ids) - 1
        if n < MIN_TOKENS:
            return {"tokens": n, "too_short": True}

        def log_softmax(model):
            logits = model(mx.array([ids]))[0, :-1, :].astype(mx.float32)
            return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        with self.lock:
            logp = log_softmax(self.model)  # performer
            ref = mx.exp(log_softmax(self.observer)) if self.observer else mx.exp(logp)
            target = mx.array(ids[1:])
            tok_logp = mx.take_along_axis(logp, target[:, None], axis=-1)[:, 0]
            mu = mx.sum(ref * logp, axis=-1)  # E_observer[log p_performer]
            var = mx.sum(ref * logp * logp, axis=-1) - mu * mu
            stats = mx.stack([mx.sum(tok_logp), mx.sum(mu), mx.sum(var)])
            mx.eval(stats)
            del logp, ref, tok_logp, mu, var
            mx.clear_cache()  # don't keep per-shape buffers pinned between texts
        sum_logp, sum_mu, sum_var = (float(x) for x in stats.tolist())
        log_ppl = -sum_logp / n
        x_ppl = -sum_mu / n
        return {
            "tokens": n,
            "log_ppl": log_ppl,
            "x_ppl": x_ppl,
            "binoculars": log_ppl / x_ppl if x_ppl > 0 else None,
            "curvature": (sum_logp - sum_mu) / math.sqrt(sum_var) if sum_var > 0 else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--performer", "--model", required=True, help="MLX model path or Hugging Face repo (instruction-tuned)")
    ap.add_argument("--observer", help="optional base model sharing the performer's tokenizer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/aidetector-scorer.sqlite"))
    ap.add_argument("--cache-limit-gb", type=float, default=2.0, help="cap on MLX's GPU buffer cache (pinned memory)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    expand = lambda p: os.path.expanduser(p) if p and p.startswith("~") else p
    scorer = Scorer(expand(args.performer), expand(args.observer), args.cache, args.cache_limit_gb)
    scorer.score("Warm up the model so the first real request isn't slow.")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"model": scorer.model_id, "max_tokens": MAX_TOKENS})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, {"model": scorer.model_id, "results": [scorer.score(t) for t in body["texts"]]})
            except Exception as e:  # report bad input to the caller instead of dropping the connection
                self._send(400, {"error": str(e)})

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    print(f"scoring with {scorer.model_id} on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
