from __future__ import annotations

import argparse
from pathlib import Path

from llm_word_roulette.baselines import save_baseline_distributions
from llm_word_roulette.corpus import default_corpus, get_corpus
from llm_word_roulette.metrics import compute_metrics
from llm_word_roulette.plots import make_plots
from llm_word_roulette.salience import compute_salience
from llm_word_roulette.text import prepare_corpus
from llm_word_roulette.website import export_website_data


def _resolve_corpus(corpus_arg: str | None) -> str:
    return corpus_arg or default_corpus()


def cmd_prepare(corpus_id: str) -> None:
    corpus = get_corpus(corpus_id)
    stats = prepare_corpus(corpus)
    compute_salience(corpus)
    print(f"[prepare] corpus={corpus_id} stats={stats}")


def cmd_baselines(corpus_id: str) -> None:
    corpus = get_corpus(corpus_id)
    run_ids = save_baseline_distributions(corpus)
    print(f"[baselines] corpus={corpus_id} run_ids={run_ids}")


def cmd_metrics(corpus_id: str) -> None:
    corpus = get_corpus(corpus_id)
    df = compute_metrics(corpus)
    print(f"[metrics] rows={len(df)} file={Path.cwd() / 'results' / 'metrics.parquet'}")


def cmd_plots(corpus_id: str) -> None:
    corpus = get_corpus(corpus_id)
    files = make_plots(corpus)
    print(f"[plots] corpus={corpus_id} files={[str(x) for x in files]}")


def cmd_web(corpus_id: str) -> None:
    corpus = get_corpus(corpus_id)
    out = export_website_data(corpus)
    print(f"[web] exported {out}")


def cmd_run_all(corpus_id: str) -> None:
    cmd_prepare(corpus_id)
    cmd_baselines(corpus_id)
    cmd_metrics(corpus_id)
    cmd_plots(corpus_id)
    cmd_web(corpus_id)


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-word-roulette")
    parser.add_argument("--corpus", default=None, help="Corpus id (default from configs/experiment.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["prepare", "baselines", "metrics", "plots", "web", "run-all"]:
        sub.add_parser(name)

    args = parser.parse_args()
    corpus_id = _resolve_corpus(args.corpus)

    if args.command == "prepare":
        cmd_prepare(corpus_id)
    elif args.command == "baselines":
        cmd_baselines(corpus_id)
    elif args.command == "metrics":
        cmd_metrics(corpus_id)
    elif args.command == "plots":
        cmd_plots(corpus_id)
    elif args.command == "web":
        cmd_web(corpus_id)
    elif args.command == "run-all":
        cmd_run_all(corpus_id)


if __name__ == "__main__":
    main()
