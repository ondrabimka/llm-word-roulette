from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_word_roulette.corpus import Corpus, corpus_paths
from llm_word_roulette.utils import ensure_dir, write_json


def _load_distribution(paths: dict[str, Path], name: str) -> pd.DataFrame:
    return pd.read_parquet(paths["distributions_dir"] / f"baseline-{name}.parquet")


def export_website_data(corpus: Corpus) -> Path:
    paths = corpus_paths(corpus.id)
    freq = _load_distribution(paths, "p_freq")
    typ = _load_distribution(paths, "p_type")
    sal = _load_distribution(paths, "p_salience")
    metrics = pd.read_parquet(Path.cwd() / "results" / "metrics.parquet")

    ollama_dirs = [
        Path.cwd() / "results" / "ollama_runs" / corpus.id,
        Path.cwd() / "docs" / "data" / "ollama_runs" / corpus.id,
    ]
    by_run_id: dict[str, dict] = {}
    for ollama_dir in ollama_dirs:
        if not ollama_dir.exists():
            continue
        for file in sorted(ollama_dir.glob("*.json")):
            run = json.loads(file.read_text(encoding="utf-8"))
            if "run_id" in run:
                by_run_id[run["run_id"]] = run
    ollama_runs = [by_run_id[k] for k in sorted(by_run_id)]

    payload = {
        "corpus": corpus.id,
        "name": corpus.name,
        "rank_curves": {
            "P_freq": freq["p_model"].sort_values(ascending=False).tolist(),
            "P_type": typ["p_model"].sort_values(ascending=False).tolist(),
            "P_salience": sal["p_model"].sort_values(ascending=False).tolist(),
        },
        "top20": {
            "P_freq": freq.head(20)[["word", "p_model"]].to_dict(orient="records"),
            "P_type": typ.head(20)[["word", "p_model"]].to_dict(orient="records"),
            "P_salience": sal.head(20)[["word", "p_model"]].to_dict(orient="records"),
        },
        "metrics": metrics.to_dict(orient="records"),
        "ollama_runs": ollama_runs,
    }

    out = Path.cwd() / "docs" / "data" / "summary.json"
    ensure_dir(out.parent)
    write_json(out, payload)
    return out
