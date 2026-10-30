from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import regex

from llm_word_roulette.cleaners import run_cleaner
from llm_word_roulette.corpus import Corpus, corpus_paths
from llm_word_roulette.utils import ensure_dir, sha256_file

WORD_RE = regex.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", flags=regex.VERSION1)


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def _read_checksum_file(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("sha256="):
            return line.split("=", 1)[1].strip()
    return None


def _write_checksum_file(path: Path, digest: str) -> None:
    ensure_dir(path.parent)
    path.write_text(f"sha256={digest}\n", encoding="utf-8")


def download_and_verify(corpus: Corpus) -> Path:
    paths = corpus_paths(corpus.id)
    ensure_dir(paths["raw_dir"])
    source = paths["source"]
    checksums = paths["checksums"]

    if not source.exists():
        with urlopen(corpus.url) as response:
            source.write_bytes(response.read())

    digest = sha256_file(source)
    stored = _read_checksum_file(checksums)
    if stored is None:
        _write_checksum_file(checksums, digest)
    elif stored != digest:
        raise ValueError(
            f"Checksum mismatch for {source}. Expected {stored}, got {digest}. "
            "Delete the file and re-run prepare if source changed intentionally."
        )

    return source


def prepare_corpus(corpus: Corpus) -> dict[str, int]:
    source = download_and_verify(corpus)
    raw_text = source.read_text(encoding="utf-8")
    clean_text, clean_stats = run_cleaner(corpus.cleaner, raw_text)

    paths = corpus_paths(corpus.id)
    ensure_dir(paths["processed_dir"])
    paths["clean"].write_text(clean_text, encoding="utf-8")

    words = tokenize_words(clean_text)
    counts = Counter(words)
    rows = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    total = sum(counts.values())

    df = pd.DataFrame(rows, columns=["word", "count"])
    df["rank"] = range(1, len(df) + 1)
    df["freq"] = df["count"] / total
    df = df[["word", "count", "rank", "freq"]]
    df.to_parquet(paths["counts"], index=False)

    return {
        **clean_stats,
        "total_tokens": int(total),
        "n_types": int(len(df)),
        "n_hapax": int((df["count"] == 1).sum()),
    }
