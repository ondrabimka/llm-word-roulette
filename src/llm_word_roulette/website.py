from __future__ import annotations

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
    }

    out = Path.cwd() / "docs" / "data" / "summary.json"
    ensure_dir(out.parent)
    write_json(out, payload)
    return out
