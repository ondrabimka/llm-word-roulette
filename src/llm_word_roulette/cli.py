from __future__ import annotations

import argparse
from pathlib import Path

from llm_word_roulette.baselines import save_baseline_distributions
from llm_word_roulette.corpus import default_corpus, get_corpus
from llm_word_roulette.metrics import compute_metrics
from llm_word_roulette.ollama_sampling import sample_ollama_distribution
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


def cmd_sample_ollama(
    corpus_id: str,
    model: str,
    n: int,
    template_id: str | None,
    pull: bool,
    rag_mode: str,
    rag_chunk_words: int,
    rag_overlap_words: int,
    rag_seed: int,
) -> None:
    corpus = get_corpus(corpus_id)
    summary = sample_ollama_distribution(
        corpus=corpus,
        model=model,
        n=n,
        template_id=template_id,
        pull=pull,
        rag_mode=rag_mode,
        rag_chunk_words=rag_chunk_words,
        rag_overlap_words=rag_overlap_words,
        rag_seed=rag_seed,
    )
    print(
        "[sample-ollama] "
        f"run_id={summary['run_id']} rag_mode={summary['rag_mode']} js_freq={summary['js_freq']:.4f} "
        f"js_type={summary['js_type']:.4f} js_salience={summary['js_salience']:.4f} "
        f"oov_rate={summary['oov_rate_sampled']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-word-roulette")
    parser.add_argument("--corpus", default=None, help="Corpus id (default from configs/experiment.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["prepare", "baselines", "metrics", "plots", "web", "run-all"]:
        sub.add_parser(name)
    sample_ollama = sub.add_parser("sample-ollama")
    sample_ollama.add_argument("--model", required=True, help="Ollama model name, e.g. qwen3:0.6b")
    sample_ollama.add_argument("--n", type=int, default=300, help="Number of samples")
    sample_ollama.add_argument("--template", default=None, help="Prompt template id from configs/prompts/*.yaml")
    sample_ollama.add_argument("--no-pull", action="store_true", help="Do not pull model before sampling")
    sample_ollama.add_argument(
        "--rag-mode",
        default="none",
        choices=["none", "random"],
        help="RAG context mode. 'random' prepends random corpus chunks.",
    )
    sample_ollama.add_argument("--rag-chunk-words", type=int, default=150, help="RAG chunk size in words")
    sample_ollama.add_argument("--rag-overlap-words", type=int, default=30, help="RAG chunk overlap in words")
    sample_ollama.add_argument("--rag-seed", type=int, default=42, help="Seed for random RAG chunk selection")

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
    elif args.command == "sample-ollama":
        cmd_sample_ollama(
            corpus_id,
            model=args.model,
            n=args.n,
            template_id=args.template,
            pull=not args.no_pull,
            rag_mode=args.rag_mode,
            rag_chunk_words=args.rag_chunk_words,
            rag_overlap_words=args.rag_overlap_words,
            rag_seed=args.rag_seed,
        )


if __name__ == "__main__":
    main()
