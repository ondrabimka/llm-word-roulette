#!/usr/bin/env sh
set -eu

python -m llm_word_roulette run-all
python -m llm_word_roulette web

cd /app/docs
python -m http.server 8000
