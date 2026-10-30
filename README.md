# llm-word-roulette

**What does a language model sample when asked for "a random word from Shakespeare"?**

This repository measures the distribution of "random" words produced by language models, with and without retrieval-augmented context (RAG), and compares those distributions to the actual word distribution of a reference text. The first (and default) corpus is the Tiny Shakespeare dataset; other texts can be added later as new corpus entries without code changes (§3.1).

Models are reached through pluggable backends: local Hugging Face `transformers`, local **Ollama**, and the **OpenAI**, **Azure OpenAI** and **Anthropic** APIs. Because these expose different amounts of probability information, each backend declares which measurement modes it supports (exact scoring, top-logprobs, sampling); see §5.2.

> This README is a build specification. It is written so a coding agent can implement the project end to end. Sections marked **MUST** are requirements; **SHOULD** are strong defaults; **MAY** are optional extensions.

---

## Quick start (implemented)

```bash
pip install -e ".[dev]"
python -m llm_word_roulette run-all
python -m llm_word_roulette sample-ollama --model qwen3:0.6b --n 250
```

This generates:

- cleaned corpus and counts in `data/processed/{corpus}/`
- baseline distributions in `results/distributions/{corpus}/`
- metrics in `results/metrics.parquet`
- figures in `results/figures/{corpus}/`
- interactive dashboard data in `docs/data/summary.json`

Open `docs/index.html` locally or enable GitHub Pages on the repository. A ready workflow is included at `.github/workflows/pages.yml`.

### Local deploy on `develop` with Docker

```bash
git checkout develop
docker compose up --build dashboard
```

Then open: `http://localhost:8000`

Optional (if you want Ollama in Docker too):

```bash
docker compose --profile ollama up -d ollama
```

---

## 1. Motivation

LLMs asked for a "random number between 1 and 100" are strongly biased (e.g. 37, 42, 47, 73), mirroring human bias. This project asks the equivalent question for words, against a concrete text with a known distribution.

"A random word from a book" has two honest interpretations, which give two reference distributions:

1. **Random position**: point at a random spot in the text. Words appear proportionally to their frequency, so the reference follows Zipf's law ("the", "and", "I" dominate).
2. **Random vocabulary entry**: pick uniformly among distinct words. Rare words dominate, since most distinct words occur once or twice.

The working hypothesis is that models do **neither** and instead favour *salient, characteristic* words ("love", "thou", "king", "death").

## 2. Research questions and hypotheses

| ID | Question | Hypothesis |
|----|----------|------------|
| Q1 | How does each model's random-word distribution compare with the text's frequency distribution and the uniform-over-types distribution? | Closer to neither; low mass on stop words, high mass on salient content words. |
| Q2 | Does the model favour words characteristic of Shakespeare relative to general English? | Model probability correlates with salience (Shakespeare frequency vs general-English frequency). |
| Q3 | How do the distributions change with model size and with training progress? | Larger / later models concentrate more on salient words; the shape of the rank-probability curve changes. |
| Q4 | What happens when relevant text is retrieved into the context (RAG)? | Probability mass shifts toward words present in the context ("copy effect"), but the salience bias persists. |
| Q5 | Does the retriever itself introduce bias (which passages get retrieved for a "random word" query)? | Yes: retrieval concentrates on a small set of chunks rather than covering the corpus uniformly. |
| Q6 | What is the Zipf exponent of each model's distribution vs the text's? | Differs from the text; reported per model and condition. |
| Q7 | How do instruction-tuned chat models (local via Ollama, and OpenAI, Azure OpenAI, Anthropic) differ from base models, and does explicitly asking for frequency-weighted sampling help? | Chat models are more concentrated (lower entropy, a few favourite words) than base models; the explicit instruction changes little. |

## 3. Data

### 3.1 Corpora (MUST, `configs/corpora.yaml`)

The project is built around a **corpus abstraction** so that further texts can be added later. Only Shakespeare is implemented and run at first, but nothing in the pipeline may hard-code it: every path, prompt, metric row and run id is keyed by `corpus`.

```yaml
- id: shakespeare                      # used in paths and run ids
  name: "Shakespeare"                  # human-readable source name used in prompts
  source_phrase: "the works of Shakespeare"   # inserted into prompt templates
  language: en                         # selects stop words and the wordfreq language
  url: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  sha256: null                         # filled on first download, then verified
  cleaner: shakespeare_dialogue        # name of a registered cleaning function
  extra_stopwords: [thou, thee, thy, thine, art, hath, doth, ye]   # reported separately (§3.3)
  license: public domain
```

- Default corpus: `shakespeare` (Tiny Shakespeare, Karpathy char-rnn, about 1.1 MB of plain text).
- **MUST** download each corpus once to `data/raw/{corpus}/source.txt` and record its SHA-256 in `data/raw/{corpus}/CHECKSUMS`. All later runs verify the checksum.
- Adding a corpus means: a new entry in `corpora.yaml`, optionally a new cleaner registered in `src/llm_word_roulette/cleaners.py`, and (for a new language) a stop-word list. A `generic_plaintext` cleaner (strip Project Gutenberg headers and footers, normalize whitespace) **MUST** exist so most texts need no new code.
- Every CLI command takes `--corpus`; the default comes from `experiment.yaml: default_corpus` (initially `shakespeare`), not from code.

### 3.2 Cleaning (MUST, `src/llm_word_roulette/cleaners.py` and `text.py`)

The `shakespeare_dialogue` cleaner handles the Tiny Shakespeare format. The raw file is dialogue with speaker labels, e.g.:

```
First Citizen:
Before we proceed any further, hear me speak.
```

Rules:

1. Remove speaker-label lines: a line that consists only of a name followed by `:` (regex roughly `^[A-Za-z][A-Za-z '\-]*:\s*$`). Log how many lines were removed.
2. Keep the remaining text with line breaks (needed for chunking in RAG).
3. Word tokenization for counting (shared by all corpora, in `text.py`): lowercase, extract words with a Unicode-aware pattern (letters only, e.g. `[^\W\d_]+(?:'[^\W\d_]+)?` with the `regex` module), so later non-English corpora with diacritics (e.g. Czech "příliš") work unchanged. Keeps "o'er"; "'tis" becoming `tis` is acceptable; document the choice. No stemming or lemmatization.
4. Output:
   - `data/processed/{corpus}/clean.txt`: cleaned text.
   - `data/processed/{corpus}/word_counts.parquet`: columns `word, count, rank, freq` (freq = count / total tokens), sorted by count descending, ties broken alphabetically.
5. Log: total word tokens, number of distinct words (types), number of hapax legomena (count = 1).

### 3.3 Stop-word lists

- **MUST** use fixed, versioned lists per language stored in `data/stopwords/{language}.txt` (for English, e.g. the NLTK English list).
- Corpus-specific additions come from `extra_stopwords` in `corpora.yaml` (for Shakespeare the Early Modern forms `thou, thee, thy, thine, art, hath, doth, ye`). These **MUST** be reported separately, since they are both stop words and markers of the source's style ("Shakespeareness"). Report stop-word mass both with and without them.

### 3.4 General-language reference

- **MUST** use the `wordfreq` library (`zipf_frequency(word, language)`) as the general-language frequency reference for salience, with `language` taken from the corpus entry.
- Salience of word *w*: `salience(w) = log10(freq_corpus(w) * 1e9) - zipf_frequency(w, language)`. Words absent from wordfreq get `zipf_frequency = 0` (wordfreq's default); flag them in a boolean column `not_in_wordfreq`.
- Output: `data/processed/{corpus}/salience.parquet`.

## 4. Reference distributions (MUST, `src/llm_word_roulette/baselines.py`)

Over the vocabulary *V* (distinct lowercase words of the cleaned text):

| Name | Definition |
|------|------------|
| `P_freq` | count(w) / total tokens. Random-position sampling. |
| `P_type` | 1 / \|V\|. Uniform over distinct words. |
| `P_salience` | softmax of salience over V (temperature 1). A reference for "picks characteristic words". |
| `P_context` | (RAG conditions only) word frequencies within the context actually shown to the model, zero elsewhere, smoothed with add-1e-9 then renormalized. |

Also provide samplers for `P_freq` and `P_type` (used in tests and plots).

## 5. Models and backends

### 5.1 Two tracks

| Track | Models | Prompting | Main measurement |
|-------|--------|-----------|------------------|
| **Base track** | Base (non-instruction-tuned) models, local only (Hugging Face, Ollama base/`text` tags) | Completion prompts (§6.2) | Exact scoring of the whole vocabulary (HF); top-logprobs + sampling (Ollama) |
| **Chat track** | Instruction-tuned / chat models: Ollama instruct tags, OpenAI, Azure OpenAI, Anthropic | Chat prompts (§6.6) | Sampling, plus top-logprobs where the API offers them |

Results from the two tracks are reported side by side but **MUST NOT** be pooled: the prompts and the measurement precision differ.

### 5.2 Backend capability matrix

Every backend implements a common interface (`src/llm_word_roulette/backends/base.py`) and declares its capabilities. The pipeline picks the most precise mode a backend supports and refuses unsupported modes with a clear error.

| Backend | `score` (exact P of any word) | `top_logprobs` (partial distribution) | `sample` | Notes |
|---------|:---:|:---:|:---:|-------|
| `hf` (transformers) | ✅ | ✅ | ✅ | Reference implementation; full control of tokens and KV cache. |
| `ollama` | ❌ | ✅ | ✅ | `logprobs` / `top_logprobs` on `/api/generate` and `/api/chat` (requires a recent Ollama; check the version at startup). Use `raw: true` on `/api/generate` for base models so no chat template is applied. Not every model returns logprobs: detect missing fields and downgrade to `sample`. |
| `openai` | ❌ | ✅ (first tokens, `top_logprobs` ≤ 20) | ✅ | Chat Completions API only (the Responses API does not return logprobs). Reasoning models do not return logprobs and may ignore `temperature`: mark them `sample`-only. |
| `azure_openai` | ❌ | ✅ (deployment-dependent) | ✅ | Same as `openai`, addressed by endpoint, API version and deployment name. Capability is probed per deployment. |
| `anthropic` | ❌ | ❌ | ✅ | The Messages API does not expose logprobs; sampling only. |

Capabilities **MUST** be verified at runtime by a cheap probe request (cached), not only trusted from the table, since provider support changes. The probe result is stored in `results/backend_capabilities.json`.

### 5.3 Backend interface (MUST)

```python
class Backend(Protocol):
    name: str
    capabilities: set[Literal["score", "top_logprobs", "sample"]]

    def score(self, prefix: str, continuations: list[str]) -> list[float]: ...
        # log P(continuation | prefix), summed over tokens

    def top_logprobs(self, prompt: Prompt, k: int, max_tokens: int) -> list[TokenStep]: ...
        # per generated position: chosen token, its logprob, top-k alternatives

    def sample(self, prompt: Prompt, n: int, temperature: float,
               max_tokens: int, stop: list[str]) -> list[str]: ...
```

`Prompt` holds either a raw completion string (base track) or a list of chat messages (chat track) plus an optional system prompt.

### 5.4 Configuration and secrets

- Credentials come only from environment variables, never from config files or logs:
  `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST` (default `http://localhost:11434`), `HF_TOKEN` (gated HF models).
- Provide `.env.example` listing these with empty values; `.env` is git-ignored.
- `configs/models.yaml` entries gain a `backend` field and backend-specific options:

```yaml
- id: gpt2
  backend: hf
  track: base
  revision: main
  dtype: float32

- id: llama3.1:8b-text-q8_0        # example tag; verify in the Ollama library
  backend: ollama
  track: base
  raw: true
  quantization: q8_0               # recorded, since quantization changes probabilities

- id: llama3.1:8b-instruct-q8_0
  backend: ollama
  track: chat

- id: gpt-4.1-mini                 # example; verify current model names
  backend: openai
  track: chat

- id: my-gpt-deployment            # Azure deployment name
  backend: azure_openai
  track: chat
  underlying_model: gpt-4.1-mini   # recorded for the report

- id: claude-sonnet-4-5            # example; verify current model ids
  backend: anthropic
  track: chat
```

Model ids for API providers change often; the agent **MUST** check each provider's current model list before running and record the exact id returned in API responses.

### 5.5 Local Hugging Face models (base track)

The agent **MUST** verify each repository id on Hugging Face before running and record the exact revision hash used.

| Family | Ids (verify) | Why |
|--------|--------------|-----|
| GPT-2 | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` | Classic baseline, small, 1024-token context. |
| Pythia | `EleutherAI/pythia-{70m,160m,410m,1b,1.4b,2.8b,6.9b,12b}` | Same data and order across sizes; intermediate checkpoints (revisions `step1000` ... `step143000`) for Q3. |
| Pythia deduped | `EleutherAI/pythia-*-deduped` | MAY: effect of training-data deduplication. |
| OLMo | a current OLMo base model (look up the id) | Fully open training data, so Shakespeare exposure can be checked. |
| Qwen base | e.g. `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-7B` | Modern base models. Confirm these are the base (not `-Instruct`) checkpoints. |
| Llama / Mistral base | e.g. `meta-llama/Llama-3.1-8B`, `mistralai/Mistral-7B-v0.1` | MAY. Llama is gated; requires `HF_TOKEN`. |

Model lists live in `configs/models.yaml`, not hard-coded. Each HF entry: `id`, `backend: hf`, `track: base`, `revision`, `dtype`, `max_context`, `batch_size`, `family`, `params`.

**Cross-backend check (SHOULD):** run at least one model both via `hf` (exact scoring) and via `ollama` (same model, GGUF quantized) on the base track. This validates the Ollama top-logprobs path against ground truth and measures how much quantization shifts the random-word distribution.

Pythia checkpoint sweep (Q3), default steps: `1000, 4000, 16000, 32000, 64000, 100000, 143000` on `pythia-410m` and `pythia-1.4b`.

## 6. Method

Three measurement modes, from most to least precise. Each run records which mode produced it (`mode` column in all results).

| Mode | Backends | Output |
|------|----------|--------|
| `score` | `hf` | Exact `P_model` over the whole vocabulary V (§6.1 to §6.4). |
| `top_logprobs` | `hf`, `ollama`, `openai`, `azure_openai` | Partial distribution over the most likely words, with the covered probability mass reported (§6.5). |
| `sample` | all | Empirical distribution from N samples, with bootstrap confidence intervals (§6.6 to §6.7). |

### 6.1 Principle of exact scoring

For a base model and a prompt, compute the probability of every word in V as the continuation. This yields the model's full distribution over V with no sampling noise.

### 6.2 Prompt templates (MUST, `configs/prompts/{corpus}.yaml`)

Prompts end **without** a trailing space; the continuation starts with a space, which matches how BPE tokenizers encode words. Each template defines a `prefix` and a `terminator`.

Templates are stored per corpus, because good wording depends on the text ("plays" fits Shakespeare, not a novel). If a corpus has no file of its own, `configs/prompts/_generic.yaml` is used, with placeholders `{name}` and `{source_phrase}` filled from `corpora.yaml` (e.g. `Here is a word chosen completely at random from {source_phrase}:`). Templates for a non-English corpus SHOULD be written in that corpus's language as well as in English, and the two reported separately.

Defaults for `shakespeare`:

| id | prefix | terminator |
|----|--------|-----------|
| `t1` | `Here is a word chosen completely at random from the works of Shakespeare:` | `\n` |
| `t2` | `Random word from Shakespeare:` | `\n` |
| `t3` | `I opened a book of Shakespeare's plays at a random page and the first word my finger landed on was` | `.` |
| `t4` | `List of random words from Shakespeare's plays:\n1.` | `\n` |
| `t5` | `Shakespeare word of the day:` | `\n` |

Results are reported per template and averaged across templates (mean of log-probabilities is NOT the right average; average the normalized distributions).

### 6.3 Scoring a word (MUST, `src/llm_word_roulette/scoring.py`)

For prompt prefix `p`, word `w`, terminator `t`:

```
continuation = " " + w + t
log P(w | p) = sum over continuation tokens of log p(token_i | p, tokens_<i)
```

Requirements:

1. **Tokenization boundary.** Tokenize `p` and `p + continuation` separately. Assert that the tokens of `p` are a prefix of the tokens of `p + continuation`. If not, log the word and handle by tokenizing the continuation alone and concatenating ids (document this in the run log).
2. **Terminator included.** Without it, "lov" would absorb probability that belongs to "love". The terminator probability is part of the score.
3. **Case variants.** For each lowercase type `w`, score both `w` and `w.capitalize()` and combine with `logsumexp`. Do not score ALL-CAPS variants.
4. **Efficiency.** Run the prompt once, cache its past key values, and batch the continuations (padding right, attention mask, gather the correct positions). Target: a 1B model scores the full vocabulary for one template in minutes on one GPU.
5. **Normalization.** `P_model(w) = exp(logP(w)) / sum over V exp(logP(v))`. Also store `Z = sum over V exp(logP(v))`, the unnormalized mass on in-text words, as `in_vocab_mass`. A low value means the model puts most probability on words that do not occur in the corpus.
6. **Numerics.** Compute in float32 for the log-softmax even if the model runs in bf16/fp16.

### 6.4 Sampling sanity check (MUST)

For each (model, template), also generate N = 1000 samples with `temperature=1.0`, `top_p=1.0`, `max_new_tokens=8`, stop at the terminator. Parse the first word.

- Estimate the out-of-vocabulary rate (fraction of sampled words not in V). This complements `in_vocab_mass`.
- Compare the empirical distribution of in-vocabulary samples with `P_model`: chi-square or JS divergence must be small. A large mismatch indicates a scoring bug.

### 6.5 Top-logprobs mode (`src/llm_word_roulette/toplogprobs.py`)

APIs return the top-k alternatives at each generated position but cannot score an arbitrary word. Build a partial distribution as follows:

1. Request one generation with `temperature=0` (or the backend's lowest allowed), `logprobs=true`, `top_logprobs=k` (k = 20 for OpenAI/Azure; configurable for Ollama, default 20), `max_tokens` of a few tokens.
2. **First-token distribution (all logprob backends, MUST):** take the top-k alternatives at the first content position. Map each token to a word prefix (strip leading space, lowercase). Tokens that are complete words in V get their probability directly; record the total covered mass `covered_mass = sum of exp(logprob)` over the top-k.
3. **Tree expansion (Ollama base track, SHOULD; requires `raw: true`):** for each top-k first token that is not yet a complete word followed by the terminator, append it to the raw prompt and request the next position's top-k. Recurse up to depth 4 or until the cumulative path probability falls below 1e-4. A word's probability is the product along its path including the terminator. This gives an exact-for-the-head, truncated distribution. Cap total requests per run (config `max_tree_requests`, default 2000) and log when the cap is hit.
4. OpenAI and Azure chat APIs cannot continue from an arbitrary partial assistant message, so they use step 2 only. Report clearly that their top-logprob distribution covers first tokens only.
5. Store `covered_mass`, depth reached and request count with each run. Metrics computed on a partial distribution are restricted to covered words and flagged `partial=True`.

### 6.6 Chat-track prompts (MUST, `configs/prompts_chat/{corpus}.yaml`)

Chat models need instructions, and their output must be parsed. Each chat template has a `system` (optional) and `user` message and asks for exactly one word. Per-corpus files and the `_generic.yaml` fallback work as in §6.2. Defaults for `shakespeare`:

| id | user message |
|----|--------------|
| `c1` | `Pick a word at random from the works of Shakespeare. Reply with only that word, nothing else.` |
| `c2` | `Give me one random word from Shakespeare's plays. Output a single word.` |
| `c3` | `Imagine opening a volume of Shakespeare at a random page and putting your finger on a word. Which word is it? Answer with the word only.` |
| `c4` | `Random Shakespeare word:` (with system prompt `Answer with exactly one word.`) |
| `c5` | `Imagine the complete text of Shakespeare's plays and pick one word position uniformly at random, so common words like "the" are picked as often as they occur. Reply with one word only.` |

`c5` explicitly asks for frequency-weighted (random-position) sampling; compare it with the others to see whether the instruction changes behaviour.

**Parsing (MUST):** strip whitespace, quotes and trailing punctuation; take the first `[A-Za-z']+` token; lowercase. Store the raw response alongside the parsed word. Record the parse-failure rate (empty or multi-word responses) per run; responses that fail to parse are excluded from the distribution but counted.

**Reasoning / thinking models:** disable visible reasoning where the API allows it, or record that it was on. Parse only the final answer.

### 6.7 Sampling mode details

- N = 2000 samples per (model, template, condition) by default (config `n_samples`). Use `temperature=1.0` and leave other sampling parameters at provider defaults unless the provider requires otherwise; record all parameters actually sent.
- Each sample is an independent request (no conversation history). Where a provider offers `n` > 1 per request (OpenAI, Azure), it MAY be used to cut cost; record it.
- Set `seed` where supported (OpenAI best-effort), but do not rely on it for reproducibility; the cached responses are the reproducible artifact.
- **Fair baselines for finite N (MUST):** divergences estimated from N samples are biased upward. For every metric computed from samples, also compute it for N samples drawn from `P_freq` and from `P_type`, and report the model's value next to these sampled baselines. Report 95% bootstrap confidence intervals (1000 resamples).
- For `hf` models, run sampling too, so the bias of the sampling estimator can be measured against the exact distribution.

## 7. Conditions

### 7.1 C0: Prior (no context)

Templates from 6.2 as they are.

### 7.2 RAG conditions

Context is inserted before the prompt prefix:

```
{context}\n\n{prefix}
```

For the chat track, the context goes into the user message, never the system prompt:

```
Here is an excerpt from {name}:

<excerpt>
{context}
</excerpt>

{chat template user message}
```

The retrieval step is identical for all backends: the same retrieved chunks (same query, retriever, k) are reused across every model, so differences come from the models only.

**Index construction (`src/llm_word_roulette/rag.py`):**

- Chunk `clean.txt` into windows of about 150 words with 30-word overlap (configurable). Store `chunk_id, start_word, end_word, text`.
- Retrievers (both MUST be implemented):
  - Dense: `sentence-transformers` with `BAAI/bge-small-en-v1.5` (default for English) or `all-MiniLM-L6-v2`; FAISS `IndexFlatIP` over normalized embeddings. The embedding model is configurable per language (a multilingual model such as `BAAI/bge-m3` for non-English corpora).
  - Sparse: BM25 (`rank_bm25`).
- Index files cached in `data/index/{corpus}/`.

**Queries (`configs/queries/{corpus}.yaml`):**

- Neutral (Shakespeare defaults): `"a random word from Shakespeare"`, `"pick any word"`, `"Shakespeare"`, `"random"`.
- Topical (MAY, for steering): `"love"`, `"war"`, `"death"`, `"king"`.

**Conditions:**

| id | Context | Purpose |
|----|---------|---------|
| `C1-rag-k{1,3}` | Top-k retrieved chunks for each neutral query | Realistic RAG. |
| `C2-random-k{1,3}` | k chunks drawn uniformly at random (average over R = 20 draws) | Control: separates "any context" from "retrieved context". |
| `C3-topical` | Top-k for topical queries | MAY: shows how query wording steers the "random" word. |

Context length budget: the total prompt (context + prefix + longest continuation) **MUST** fit in each model's context window. For GPT-2 (1024 tokens) use k ≤ 3 with 150-word chunks; truncate from the front if needed and log it.

### 7.3 Retrieval coverage (Q5)

Independently of the LLM: run every neutral query against both retrievers, plus 200 paraphrases of "give me a random word" (MAY: generated once and stored in `configs/query_paraphrases.txt`), and record which chunks are returned. Report:

- distribution of retrieval counts per chunk,
- fraction of the corpus ever retrieved,
- Gini coefficient of retrieval counts,
- position of retrieved chunks within the text (plot along the corpus).

## 8. Metrics (MUST, `src/llm_word_roulette/metrics.py`)

For each (corpus, backend, model, revision, template, condition, mode) produce one row in `results/metrics.parquet`. Rows carry `corpus`, `backend`, `track`, `mode` (`score` / `top_logprobs` / `sample`) and `partial` (true for truncated distributions). For `sample` mode, every metric has `_ci_low` and `_ci_high` columns (bootstrap) and a matching sampled-baseline value (§6.7). Metrics that need the full distribution (e.g. `in_vocab_mass`, `hapax_mass` in `top_logprobs` mode) are left empty rather than estimated.

Additional columns by mode:

| Column | Modes | Definition |
|--------|-------|------------|
| `covered_mass` | `top_logprobs` | Probability mass covered by the returned alternatives. |
| `n_samples`, `n_parsed`, `parse_failure_rate` | `sample` | Requested samples, successfully parsed one-word answers, and the failure share. |
| `n_distinct` | `sample` | Distinct words among parsed samples (compare with the sampled baselines). |
| `quantization` | `ollama` | Quantization tag of the model file. |

Core metrics:

| Metric | Definition |
|--------|------------|
| `js_freq`, `js_type`, `js_salience`, `js_context` | Jensen-Shannon divergence (base 2) between `P_model` and each reference. |
| `spearman_freq` | Spearman correlation of `P_model(w)` and `count(w)` over V. |
| `spearman_salience` | Spearman correlation of `P_model(w)` and `salience(w)`. |
| `entropy_bits` | Entropy of `P_model`. |
| `effective_vocab` | `2 ** entropy_bits`. |
| `stopword_mass` | Sum of `P_model` over stop words (with and without the corpus's `extra_stopwords`, e.g. Early Modern forms for Shakespeare). |
| `top1_word`, `top1_prob`, `top20` | Most likely words. |
| `hapax_mass` | Mass on words occurring once in the text. |
| `in_vocab_mass` | From 6.3 step 5. |
| `oov_rate_sampled` | From 6.4. |
| `zipf_alpha`, `zipf_r2` | Slope of log(prob) vs log(rank) fitted by OLS over ranks 10 to 1000 (configurable). Same fit for `P_freq` for comparison. MAY also fit with the `powerlaw` package (MLE). |
| `context_mass` | (RAG) mass on words present in the context. |
| `spearman_context` | (RAG) correlation with in-context counts, over words in the context. |

Full per-word distributions go to `results/distributions/{corpus}/{run_id}.parquet` with columns `word, logp_raw, p_model`.

## 9. Plots (MUST, `src/llm_word_roulette/plots.py`, saved to `results/figures/`)

1. **Rank-probability, log-log.** `P_freq` (the text's Zipf curve), `P_type`, and each model's `P_model` (each sorted by its own rank). One panel per model family.
2. **Model probability vs text frequency** scatter (log-log) per model, with stop words coloured.
3. **Model probability vs salience** scatter.
4. **Top-20 words** bar charts per model (C0, template-averaged).
5. **Scaling:** `js_freq`, `spearman_salience`, `stopword_mass`, `entropy_bits` vs parameter count (Pythia, GPT-2, Qwen).
6. **Training dynamics:** the same metrics vs Pythia training step.
7. **RAG effect:** C0 vs C1 vs C2 for `context_mass`, `js_freq`, `spearman_salience`.
8. **Retrieval coverage:** chunk retrieval histogram and retrieval position along the corpus.
9. **Template sensitivity:** spread of each metric across templates per model.
10. **Base vs chat:** core metrics for base-track and chat-track models side by side, with sampled baselines and confidence intervals.
11. **Provider comparison (chat track):** top-20 words per provider/model; `c5` (explicit frequency-weighted instruction) vs the other chat templates.
12. **Backend agreement:** the same model via `hf` exact scoring vs `ollama` top-logprobs (head of the distribution) and vs sampling; shows quantization and estimator effects.

Plots use matplotlib only, consistent style, labelled axes with units, and are regenerated from parquet files without re-running models. Figures are written per corpus to `results/figures/{corpus}/`; once a second corpus exists, a cross-corpus plot (MAY) compares the core metrics for the same models across texts.

## 10. Repository layout

```
llm-word-roulette/
├── README.md
├── pyproject.toml            # uv or pip; pinned versions
├── .env.example              # names of required environment variables, no values
├── configs/
│   ├── corpora.yaml          # one entry per text; only `shakespeare` at first
│   ├── models.yaml           # every model with its backend and track
│   ├── prompts/              # base-track templates: shakespeare.yaml, _generic.yaml
│   ├── prompts_chat/         # chat-track templates: shakespeare.yaml, _generic.yaml
│   ├── queries/              # RAG queries per corpus: shakespeare.yaml
│   ├── prices.yaml           # API prices, filled in by the user
│   └── experiment.yaml       # chunking, k, N samples, seeds, rank range, budgets, concurrency
├── data/
│   ├── raw/{corpus}/         # source.txt, CHECKSUMS
│   ├── processed/{corpus}/   # clean.txt, word_counts.parquet, salience.parquet
│   ├── index/{corpus}/       # FAISS + BM25 caches
│   └── stopwords/            # en.txt (more languages later)
├── src/llm_word_roulette/
│   ├── __init__.py
│   ├── cli.py                # entry point
│   ├── corpus.py             # load corpora.yaml, resolve paths, fill prompt placeholders
│   ├── cleaners.py           # registered cleaners: shakespeare_dialogue, generic_plaintext
│   ├── text.py               # download, tokenize, count
│   ├── baselines.py          # reference distributions and samplers
│   ├── salience.py           # wordfreq-based salience
│   ├── backends/
│   │   ├── base.py           # Backend protocol, Prompt, TokenStep, capability probe
│   │   ├── hf.py             # transformers: score, top_logprobs, sample
│   │   ├── ollama.py         # HTTP client for /api/generate (raw) and /api/chat
│   │   ├── openai.py         # Chat Completions
│   │   ├── azure_openai.py   # Azure OpenAI (endpoint + deployment)
│   │   ├── anthropic.py      # Messages API, sample only
│   │   └── registry.py       # backend lookup from models.yaml
│   ├── cache.py              # on-disk request/response cache (SQLite)
│   ├── budget.py             # request and cost limits, rate limiting
│   ├── parsing.py            # chat-response to single word
│   ├── scoring.py            # exact vocabulary scoring with KV cache (hf)
│   ├── toplogprobs.py        # partial distributions, tree expansion
│   ├── sampling.py           # sampling mode and sanity-check sampling
│   ├── rag.py                # chunking, retrievers, context building
│   ├── metrics.py
│   ├── plots.py
│   └── utils.py              # seeding, logging, run ids, hashing
├── cache/                    # API and Ollama response cache (git-ignored)
├── results/
│   ├── backend_capabilities.json
│   ├── metrics.parquet       # all corpora, with a `corpus` column
│   ├── distributions/{corpus}/
│   ├── samples/{corpus}/
│   ├── retrieval/{corpus}/
│   └── figures/{corpus}/
├── notebooks/
│   └── 01_explore.ipynb      # reads results only
└── tests/
```

## 11. Command-line interface

```
python -m llm_word_roulette prepare                 # download, verify checksum, clean, counts, salience
python -m llm_word_roulette baselines               # reference distributions
python -m llm_word_roulette index                   # build FAISS and BM25 indexes
python -m llm_word_roulette probe                   # check reachability and capabilities of every configured backend/model
python -m llm_word_roulette score       --model gpt2 --condition C0 [--template t1] [--revision main]
python -m llm_word_roulette toplogprobs --model llama3.1:8b-text-q8_0 --condition C0
python -m llm_word_roulette sample      --model claude-sonnet-4-5 --condition C0 --n 2000
python -m llm_word_roulette retrieval-coverage
python -m llm_word_roulette metrics                 # (re)compute metrics from saved distributions
python -m llm_word_roulette plots
python -m llm_word_roulette run-all --models configs/models.yaml [--backend openai] [--track chat] [--dry-run]
```

- Every command accepts `--corpus` (default from `experiment.yaml: default_corpus`); `run-all` also accepts `--corpus all`.
- The backend is taken from `models.yaml`; `--backend` and `--track` only filter which models run.
- `--dry-run` prints every planned request count and the estimated API cost per model, then exits without calling anything.
- Every run gets a deterministic `run_id` = hash of (corpus, corpus checksum, backend, model id, revision, mode, template, condition, query, k, config hash).
- Runs are **resumable**: skip a run whose output exists and whose config hash matches.
- Log to `results/runs.jsonl`: run_id, timestamps, backend, exact model id returned by the provider, Ollama version and quantization, GPU, library versions, model revision hash, request parameters actually sent, warnings (tokenization fallbacks, truncations, capability downgrades, budget stops).

### 11.1 Caching, rate limits and cost control (MUST)

- **Cache every remote call** (Ollama and APIs) in `cache/requests.sqlite`, keyed by a hash of (backend, model, full request body, sample index). Re-running an experiment reads from the cache. The cache is the reproducibility artifact for API results; MAY export it to `results/samples/` as parquet.
- **Concurrency:** async requests with a per-backend semaphore (`experiment.yaml: concurrency`, defaults: ollama 2, openai 8, azure_openai 8, anthropic 4).
- **Retries:** exponential backoff with jitter on 429 and 5xx, honouring `Retry-After`; give up after 6 attempts and record the failure.
- **Budgets:** `experiment.yaml` sets `max_requests_per_run` and `max_cost_usd_total`. Cost is estimated from token usage returned by each API and a price table in `configs/prices.yaml` (to be filled in by the user; the agent MUST NOT invent prices). Stop cleanly when a budget is reached, keeping completed work.
- Never log prompts together with API keys; redact the `Authorization` and `x-api-key` headers in any debug logs.

## 12. Tests (MUST, `pytest`)

1. Cleaning removes speaker labels and keeps dialogue (fixture with a few lines).
2. Word counts sum to total tokens; ranks are consecutive.
3. `P_freq`, `P_type`, `P_model` each sum to 1 within 1e-6.
4. Tokenization boundary check passes for a sample of words on GPT-2 and one Pythia tokenizer; fallback path is exercised by a crafted case.
5. Scoring with KV cache equals scoring without cache (max abs diff in log-prob < 1e-3) on `gpt2` for 50 words.
6. Scoring `P_model` vs sampling: JS divergence below a threshold on `gpt2` (small N acceptable in CI, marked `slow`).
7. JS divergence of a distribution with itself is 0; with a disjoint one is 1.
8. Zipf fit on synthetic data with known exponent recovers it within 0.05.
9. RAG context respects the model's context window.
10. **Backends without network:** each remote backend is tested against recorded HTTP fixtures (e.g. `respx` or `vcrpy`), covering success, missing logprobs, 429 with `Retry-After`, and malformed responses. Default `pytest` makes no network calls.
11. Capability downgrade: a backend whose probe returns no logprobs is run in `sample` mode and the downgrade is logged.
12. Parsing: a table of raw chat responses (`"Love."`, `" 'Tis"`, `"The word is: crown"`, `"I can't choose randomly, but..."`, empty) maps to the expected word or a parse failure.
13. Cache: a repeated request is served from the cache without calling the backend.
14. Tree expansion on a mocked backend with a known distribution reproduces it exactly within the depth limit.
15. Integration tests MAY run against a live local Ollama (`pytest -m ollama`) and live APIs (`pytest -m live`, never in CI by default).
16. **Corpus abstraction:** a tiny second corpus fixture (a few paragraphs, `generic_plaintext` cleaner, `_generic` prompts) runs through `prepare`, `baselines` and `score` on `gpt2` with no code changes, and its outputs land in its own `{corpus}` directories. A grep-style test fails if the string "shakespeare" appears in `src/` outside `cleaners.py`.
17. Tokenization handles non-ASCII letters (e.g. "příliš žluťoučký" yields two words).

## 13. Milestones and acceptance criteria

Build in this order; each milestone ends with tests passing and a short entry in `CHANGELOG.md`.

1. **Data.** Corpus config and `prepare` work for `shakespeare` (and the test fixture corpus); counts, Zipf plot of the text itself (plot 1 with only `P_freq` and `P_type`).
2. **Scoring on GPT-2.** C0 for all templates on `gpt2`; sanity check (6.4) passes; top-20 words printed.
3. **Metrics and plots** for GPT-2 family.
4. **Pythia size sweep**, then **checkpoint sweep**.
5. **RAG**: index, retrieval coverage report, C1 and C2 on GPT-2 and Pythia.
6. **Remaining HF models** (Qwen, OLMo, optional Llama/Mistral).
7. **Backend layer**: protocol, cache, budgets, `probe` command, mocked tests. Refactor `hf` to sit behind the same interface without changing its results (rerun milestone 2 and compare).
8. **Ollama**: base track with `raw: true` and tree expansion; chat track with an instruct tag; cross-backend check against `hf` on the same model.
9. **APIs**: OpenAI, then Azure OpenAI, then Anthropic, each first with `--dry-run`, then a small run (N = 100) before the full run.
10. **Write-up**: `results/REPORT.md` answering Q1 to Q7 with the figures, one paragraph each, plus limitations.

## 14. Compute notes

- Vocabulary size is on the order of 10k to 15k types (log the exact number); with two case variants and five templates that is ~150k short continuations per model per condition. With prompt KV caching and batching this is cheap for models up to ~1.5B and feasible on a single 24 GB GPU for 7B in bf16. 12B needs ~24 GB in bf16 or 8-bit loading (document if quantized, since it changes probabilities slightly).
- RAG conditions repeat scoring per (query, k) and per random draw; budget accordingly or reduce R for large models.
- CPU-only MUST work for `gpt2` and `pythia-70m` (slow but usable for development).
- **API volume:** chat track C0 is 5 templates × N = 2000 = 10,000 short requests per model; RAG multiplies this by the number of (query, k) combinations. Always run `--dry-run` first and start RAG runs with a reduced N.
- **Ollama:** tree expansion costs up to `max_tree_requests` calls per (model, template, condition); keep `concurrency` low on a single GPU and set `keep_alive` so the model stays loaded.

## 15. Pitfalls to handle explicitly

- **Trailing spaces in prompts** distort BPE probabilities; prompts end without a space and continuations start with one (6.2).
- **Prefix collisions** ("lov" vs "love") are handled by the terminator (6.3).
- **Speaker labels** would dominate counts if not removed (3.2).
- **Early Modern forms** ("thou", "thy") are function words but also the most "Shakespearean"; report both stop-word variants.
- **Training-data contamination**: Shakespeare is almost certainly in every model's training data. That is part of what is measured, not a flaw, but note it in the report. For OLMo, MAY check exposure using its open data tools.
- **Template sensitivity** is large for base models; never report a single template alone.
- **Instruction-tuned models** never go through completion-style scoring; they belong to the chat track.
- **Chat templates in Ollama:** `/api/generate` applies the model's template unless `raw: true`. Base-track runs MUST use raw mode; chat-track runs MUST NOT.
- **Quantization:** Ollama models are usually quantized, which shifts probabilities. Always record the quantization tag and do not compare an Ollama model to an HF model as if they were identical.
- **Silent provider changes:** API models, defaults and logprob support change over time. Record exact model ids and dates; treat API results as a snapshot.
- **Refusals and chatter:** chat models sometimes explain that they cannot be random, or answer with a sentence. Count these as parse failures, report the rate, and inspect examples rather than tuning the parser until they disappear.
- **Provider-side caching** of identical prompts does not change sampling, but identical requests MUST still be sent as independent requests (distinct cache keys via sample index) so they are not collapsed by this project's own cache.

## 16. Optional extensions (MAY)

- **More backends**: vLLM or llama.cpp server (both can return prompt logprobs, which would allow exact scoring for GGUF models), Google Gemini (offers logprobs on some models).
- **String Seed of Thought**: rerun the chat track with the SSoT prompting technique and measure how far it moves the distribution toward `P_freq` or `P_type`.
- **Indexed-passage condition**: number the words of a passage (`[1] Before [2] we [3] proceed ...`), prompt for a random index, map indices back to words. Tests whether number bias (37, 73) leaks into word choice: shuffle the passage and check whether the favourite word or the favourite index stays fixed.
- **Further corpora** (the next planned step after Shakespeare; only config, prompts and queries should be needed): a Czech public-domain classic (e.g. Čapek or Němcová from a public-domain source) to test language effects, with prompts both in Czech and in English; a famous English novel (e.g. *Moby-Dick* or *Pride and Prejudice* from Project Gutenberg) to compare a play with prose; and a text the models are unlikely to have memorized, which separates "knows the book" from "knows what random words look like".
- **Temperature**: apply temperature to the scored log-probabilities analytically (`p^(1/T)` renormalized) and report how metrics change with T, without re-running models.
- **Heaps' law**: vocabulary growth of sampled word sequences vs the text.

## 17. Related work

- Coronado-Blázquez (2025), *Deterministic or probabilistic? The psychology of LLMs as random number generators*. arXiv:2502.19965.
- S. Anand, LLM random number explorer: https://sanand0.github.io/llmrandom/
- *Failure to Mix: Large language models struggle to answer according to desired probability distributions*. arXiv:2511.14630.
- Sakana AI, *String Seed of Thought*: https://pub.sakana.ai/ssot/
- Springboards, *You can't ask an LLM to be "more random"* (random-word experiment).
- *Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task*. arXiv:2505.16164 (Zipf exponents of LLM word production vs humans).
- Mikhaylovskiy (2025), *Zipf's and Heaps' Laws for Tokens and LLM-generated Texts*, Findings of EMNLP 2025.
- *How Do LLM-Generated Texts Impact Term-Based Retrieval Models?* arXiv:2508.17715 (two-regime Zipf fits).
- Berman (2025), *Random Text, Zipf's Law, Critical Length, and Implications for Large Language Models*. arXiv:2511.17575 (null models; caution against over-interpreting Zipf fits).
- Piantadosi (2014), *Zipf's word frequency law in natural language: A critical review and future directions*.
- Speer, `wordfreq`: https://github.com/rspeer/wordfreq

## 18. License

Code: MIT (to be confirmed by the repository owner). Tiny Shakespeare text is public domain; model weights are subject to their own licenses.
