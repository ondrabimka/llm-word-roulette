from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_word_roulette.utils import repo_root


@dataclass(frozen=True)
class Corpus:
    id: str
    name: str
    source_phrase: str
    language: str
    url: str
    sha256: str | None
    cleaner: str
    extra_stopwords: list[str]
    license: str


def load_corpora() -> dict[str, Corpus]:
    path = repo_root() / "configs" / "corpora.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    corpora: dict[str, Corpus] = {}
    for row in data:
        corpora[row["id"]] = Corpus(
            id=row["id"],
            name=row["name"],
            source_phrase=row["source_phrase"],
            language=row["language"],
            url=row["url"],
            sha256=row.get("sha256"),
            cleaner=row["cleaner"],
            extra_stopwords=list(row.get("extra_stopwords", [])),
            license=row.get("license", ""),
        )
    return corpora


def get_corpus(corpus_id: str) -> Corpus:
    corpora = load_corpora()
    if corpus_id not in corpora:
        raise KeyError(f"Unknown corpus '{corpus_id}'. Available: {', '.join(sorted(corpora))}")
    return corpora[corpus_id]


def default_corpus() -> str:
    path = repo_root() / "configs" / "experiment.yaml"
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["default_corpus"]


def corpus_paths(corpus_id: str) -> dict[str, Path]:
    root = repo_root()
    return {
        "raw_dir": root / "data" / "raw" / corpus_id,
        "source": root / "data" / "raw" / corpus_id / "source.txt",
        "checksums": root / "data" / "raw" / corpus_id / "CHECKSUMS",
        "processed_dir": root / "data" / "processed" / corpus_id,
        "clean": root / "data" / "processed" / corpus_id / "clean.txt",
        "counts": root / "data" / "processed" / corpus_id / "word_counts.parquet",
        "salience": root / "data" / "processed" / corpus_id / "salience.parquet",
        "figures_dir": root / "results" / "figures" / corpus_id,
        "distributions_dir": root / "results" / "distributions" / corpus_id,
    }
