from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from llm_word_roulette.corpus import Corpus, corpus_paths
from llm_word_roulette.utils import ensure_dir


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _zipf_fit(probs: np.ndarray, rmin: int = 10, rmax: int = 1000) -> tuple[float, float]:
    order = np.sort(probs)[::-1]
    max_rank = min(len(order), rmax)
    if max_rank <= rmin:
        return float("nan"), float("nan")
    ranks = np.arange(1, max_rank + 1)
    y = np.log(order[:max_rank])
    x = np.log(ranks)
    x = x[rmin - 1 :]
    y = y[rmin - 1 :]
    coeff = np.polyfit(x, y, deg=1)
    pred = coeff[0] * x + coeff[1]
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-coeff[0]), float(r2)


def _load_distribution(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)[["word", "p_model"]]


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() <= 1 or b.nunique() <= 1:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def compute_metrics(corpus: Corpus) -> pd.DataFrame:
    paths = corpus_paths(corpus.id)
    counts = pd.read_parquet(paths["counts"])[["word", "count", "freq"]]
    salience = pd.read_parquet(paths["salience"])[["word", "salience"]]

    p_freq = _load_distribution(paths["distributions_dir"] / "baseline-p_freq.parquet").set_index("word")
    p_type = _load_distribution(paths["distributions_dir"] / "baseline-p_type.parquet").set_index("word")
    p_sal = _load_distribution(paths["distributions_dir"] / "baseline-p_salience.parquet").set_index("word")

    refs = {
        "freq": p_freq["p_model"],
        "type": p_type["p_model"],
        "salience": p_sal["p_model"],
    }

    stop_path = Path.cwd() / "data" / "stopwords" / f"{corpus.language}.txt"
    stop = {x.strip().lower() for x in stop_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    extra = {x.lower() for x in corpus.extra_stopwords}

    rows: list[dict[str, object]] = []
    for name, series in refs.items():
        merged = counts.merge(salience, on="word", how="left").merge(
            series.rename("p_model"), left_on="word", right_index=True, how="left"
        )
        p = merged["p_model"].to_numpy(dtype=float)
        pf = merged["freq"].to_numpy(dtype=float)

        js_freq = jensen_shannon(p, pf)
        js_type = jensen_shannon(p, np.full_like(p, 1.0 / len(p)))
        js_sal = jensen_shannon(p, refs["salience"].reindex(merged["word"]).to_numpy(dtype=float))
        ent = entropy_bits(p)
        alpha, r2 = _zipf_fit(p)

        spearman_count = _safe_spearman(merged["p_model"], merged["count"])
        spearman_salience = _safe_spearman(merged["p_model"], merged["salience"])
        top = merged.sort_values("p_model", ascending=False).head(20)
        top1 = top.iloc[0]

        stop_mask = merged["word"].isin(stop)
        stop_with_extra_mask = merged["word"].isin(stop | extra)

        rows.append(
            {
                "corpus": corpus.id,
                "backend": "baseline",
                "track": "reference",
                "model": name,
                "mode": "score",
                "partial": False,
                "js_freq": js_freq,
                "js_type": js_type,
                "js_salience": js_sal,
                "spearman_freq": float(spearman_count),
                "spearman_salience": float(spearman_salience),
                "entropy_bits": ent,
                "effective_vocab": float(2**ent),
                "stopword_mass": float(merged.loc[stop_mask, "p_model"].sum()),
                "stopword_mass_with_extra": float(merged.loc[stop_with_extra_mask, "p_model"].sum()),
                "top1_word": str(top1["word"]),
                "top1_prob": float(top1["p_model"]),
                "top20": ",".join(top["word"].tolist()),
                "hapax_mass": float(merged.loc[merged["count"] == 1, "p_model"].sum()),
                "in_vocab_mass": 1.0,
                "zipf_alpha": alpha,
                "zipf_r2": r2,
            }
        )

    result = pd.DataFrame(rows)
    out = Path.cwd() / "results" / "metrics.parquet"
    ensure_dir(out.parent)
    result.to_parquet(out, index=False)
    return result
