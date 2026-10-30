from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_word_roulette.corpus import Corpus, corpus_paths
from llm_word_roulette.utils import ensure_dir


def _load_baseline(paths: dict[str, Path], name: str) -> pd.DataFrame:
    return pd.read_parquet(paths["distributions_dir"] / f"baseline-{name}.parquet")


def make_plots(corpus: Corpus) -> list[Path]:
    paths = corpus_paths(corpus.id)
    ensure_dir(paths["figures_dir"])

    p_freq = _load_baseline(paths, "p_freq")
    p_type = _load_baseline(paths, "p_type")
    p_sal = _load_baseline(paths, "p_salience")

    out_files: list[Path] = []

    fig1 = paths["figures_dir"] / "rank_probability_baselines.png"
    plt.figure(figsize=(10, 6))
    for label, df in [("P_freq", p_freq), ("P_type", p_type), ("P_salience", p_sal)]:
        probs = np.sort(df["p_model"].to_numpy())[::-1]
        ranks = np.arange(1, len(probs) + 1)
        plt.loglog(ranks, probs, label=label)
    plt.xlabel("Rank")
    plt.ylabel("Probability")
    plt.title(f"Rank-Probability Curves ({corpus.name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()
    out_files.append(fig1)

    fig2 = paths["figures_dir"] / "top20_baselines.png"
    plt.figure(figsize=(12, 6))
    top = p_sal.head(20).copy()
    top = top.iloc[::-1]
    plt.barh(top["word"], top["p_model"])
    plt.xlabel("Probability")
    plt.ylabel("Word")
    plt.title(f"Top 20 words by P_salience ({corpus.name})")
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()
    out_files.append(fig2)

    return out_files
