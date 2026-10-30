from __future__ import annotations

import numpy as np
import pandas as pd

from llm_word_roulette.corpus import Corpus, corpus_paths
from llm_word_roulette.utils import ensure_dir


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    ex = np.exp(shifted)
    return ex / ex.sum()


def build_baselines(corpus: Corpus) -> dict[str, pd.DataFrame]:
    paths = corpus_paths(corpus.id)
    counts = pd.read_parquet(paths["counts"])
    salience = pd.read_parquet(paths["salience"])[["word", "salience"]]
    joined = counts.merge(salience, on="word", how="left")

    p_freq = joined["freq"].to_numpy(dtype=float)
    p_type = np.full(len(joined), 1.0 / len(joined), dtype=float)
    p_salience = _softmax(joined["salience"].to_numpy(dtype=float))

    outputs: dict[str, pd.DataFrame] = {}
    for name, probs in {
        "P_freq": p_freq,
        "P_type": p_type,
        "P_salience": p_salience,
    }.items():
        df = pd.DataFrame(
            {
                "word": joined["word"],
                "logp_raw": np.log(np.clip(probs, 1e-300, None)),
                "p_model": probs,
            }
        )
        outputs[name] = df.sort_values("p_model", ascending=False).reset_index(drop=True)
    return outputs


def save_baseline_distributions(corpus: Corpus) -> list[str]:
    paths = corpus_paths(corpus.id)
    ensure_dir(paths["distributions_dir"])
    outputs = build_baselines(corpus)
    run_ids: list[str] = []
    for name, df in outputs.items():
        run_id = f"baseline-{name.lower()}"
        out_path = paths["distributions_dir"] / f"{run_id}.parquet"
        df.to_parquet(out_path, index=False)
        run_ids.append(run_id)
    return run_ids


def sample_distribution(df: pd.DataFrame, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n, p=df["p_model"].to_numpy())
    return df.loc[idx, "word"].value_counts(normalize=True)
