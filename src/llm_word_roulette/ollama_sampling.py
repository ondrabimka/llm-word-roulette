from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

from llm_word_roulette.corpus import Corpus, corpus_paths
from llm_word_roulette.metrics import jensen_shannon
from llm_word_roulette.parsing import parse_chat_word
from llm_word_roulette.utils import ensure_dir, write_json


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def _load_templates(corpus: Corpus) -> list[dict[str, str]]:
    prompts_dir = Path.cwd() / "configs" / "prompts"
    corpus_file = prompts_dir / f"{corpus.id}.yaml"
    generic_file = prompts_dir / "_generic.yaml"
    path = corpus_file if corpus_file.exists() else generic_file
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["templates"]


def _post_json(url: str, payload: dict) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Ollama connection error: {exc}") from exc


def _pull_model(ollama_host: str, model: str) -> None:
    _post_json(f"{ollama_host}/api/pull", {"name": model, "stream": False})


def _sample_once(ollama_host: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 1.0,
            "top_p": 1.0,
            "num_predict": 8,
        },
    }
    response = _post_json(f"{ollama_host}/api/generate", payload)
    return str(response.get("response", "")).strip()


def _build_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def _build_prompt(prefix: str, rag_context: str | None) -> str:
    if not rag_context:
        return prefix
    return f"{rag_context}\n\n{prefix}"


def sample_ollama_distribution(
    corpus: Corpus,
    model: str,
    n: int,
    template_id: str | None = None,
    pull: bool = True,
    rag_mode: str = "none",
    rag_chunk_words: int = 150,
    rag_overlap_words: int = 30,
    rag_seed: int = 42,
) -> dict[str, object]:
    paths = corpus_paths(corpus.id)
    counts = pd.read_parquet(paths["counts"])
    vocabulary = set(counts["word"].tolist())

    templates = _load_templates(corpus)
    template = next((t for t in templates if t["id"] == template_id), templates[0] if templates else None)
    if template is None:
        raise ValueError("No prompt templates found.")
    prefix = template["prefix"]

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if pull:
        _pull_model(ollama_host, model)

    rows: list[dict[str, object]] = []
    in_vocab_counts: dict[str, int] = {}
    parsed_count = 0
    in_vocab = 0
    rng = np.random.default_rng(rag_seed)
    clean_text = paths["clean"].read_text(encoding="utf-8") if rag_mode == "random" else ""
    chunks = _build_chunks(clean_text, rag_chunk_words, rag_overlap_words) if rag_mode == "random" else []
    if rag_mode == "random" and not chunks:
        raise ValueError("RAG random mode requested but no chunks could be built from clean text.")

    for idx in range(n):
        selected_context = None
        if rag_mode == "random":
            selected_context = str(chunks[int(rng.integers(0, len(chunks)))])
        prompt = _build_prompt(prefix, selected_context)
        raw = _sample_once(ollama_host, model, prompt)
        parsed = parse_chat_word(raw)
        is_in_vocab = bool(parsed and parsed in vocabulary)
        if parsed:
            parsed_count += 1
        if is_in_vocab and parsed is not None:
            in_vocab += 1
            in_vocab_counts[parsed] = in_vocab_counts.get(parsed, 0) + 1
        rows.append(
            {
                "sample_idx": idx,
                "raw_response": raw,
                "parsed_word": parsed,
                "in_vocab": is_in_vocab,
                "rag_mode": rag_mode,
                "rag_context": selected_context,
            }
        )

    sample_df = pd.DataFrame(rows)

    probs = counts[["word"]].copy()
    probs["count_model"] = probs["word"].map(lambda w: in_vocab_counts.get(w, 0))
    if in_vocab > 0:
        probs["p_model"] = probs["count_model"] / in_vocab
    else:
        probs["p_model"] = 0.0
    probs["logp_raw"] = probs["p_model"].map(lambda x: float("-inf") if x <= 0 else math.log(x))
    probs = probs[["word", "logp_raw", "p_model"]].sort_values("p_model", ascending=False).reset_index(drop=True)

    p_model = probs["p_model"].to_numpy(dtype=float)
    p_freq = counts["freq"].to_numpy(dtype=float)
    p_type = np.full_like(p_freq, 1.0 / len(p_freq))
    sal = pd.read_parquet(paths["salience"])[["word", "salience"]].merge(counts[["word"]], on="word", how="right")
    ex = np.exp(sal["salience"].to_numpy() - sal["salience"].to_numpy().max())
    p_sal = ex / ex.sum()

    js_freq = float(jensen_shannon(p_model, p_freq))
    js_type = float(jensen_shannon(p_model, p_type))
    js_salience = float(jensen_shannon(p_model, p_sal))

    model_slug = _slug(model)
    run_id = f"ollama-{model_slug}-sample-{template['id']}-{rag_mode}-n{n}"
    out_samples = Path.cwd() / "results" / "samples" / corpus.id / f"{run_id}.parquet"
    out_dist = paths["distributions_dir"] / f"{run_id}.parquet"
    out_stats = Path.cwd() / "results" / "ollama_runs" / corpus.id / f"{run_id}.json"
    ensure_dir(out_samples.parent)
    ensure_dir(out_dist.parent)
    ensure_dir(out_stats.parent)
    sample_df.to_parquet(out_samples, index=False)
    probs.to_parquet(out_dist, index=False)

    summary = {
        "run_id": run_id,
        "corpus": corpus.id,
        "model": model,
        "template_id": template["id"],
        "rag_mode": rag_mode,
        "rag_chunk_words": rag_chunk_words if rag_mode == "random" else None,
        "rag_overlap_words": rag_overlap_words if rag_mode == "random" else None,
        "n_samples": n,
        "n_parsed": parsed_count,
        "parse_failure_rate": (n - parsed_count) / n if n > 0 else None,
        "n_in_vocab": in_vocab,
        "oov_rate_sampled": (parsed_count - in_vocab) / parsed_count if parsed_count > 0 else None,
        "js_freq": js_freq,
        "js_type": js_type,
        "js_salience": js_salience,
        "rank_curve": probs["p_model"].sort_values(ascending=False).tolist(),
        "top20": probs.head(20)[["word", "p_model"]].to_dict(orient="records"),
        "example_contexts": sample_df["rag_context"].dropna().head(3).tolist(),
        "samples_file": str(out_samples),
        "distribution_file": str(out_dist),
    }
    write_json(out_stats, summary)
    return summary
