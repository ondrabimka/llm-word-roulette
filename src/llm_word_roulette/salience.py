from __future__ import annotations

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from llm_word_roulette.corpus import Corpus, corpus_paths


def compute_salience(corpus: Corpus) -> pd.DataFrame:
    paths = corpus_paths(corpus.id)
    counts = pd.read_parquet(paths["counts"])

    zipf = counts["word"].map(lambda w: float(zipf_frequency(w, corpus.language)))
    salience = np.log10(counts["freq"] * 1e9) - zipf

    out = counts[["word", "freq"]].copy()
    out["zipf_frequency"] = zipf
    out["salience"] = salience
    out["not_in_wordfreq"] = out["zipf_frequency"] == 0.0
    out.to_parquet(paths["salience"], index=False)
    return out
