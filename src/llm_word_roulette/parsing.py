from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z']+")


def parse_chat_word(response: str) -> str | None:
    cleaned = response.strip().strip("\"'`")
    cleaned = re.sub(r"[.,;:!?]+$", "", cleaned)
    tokens = [t.lower().lstrip("'").rstrip("'") for t in WORD_RE.findall(cleaned)]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    if ":" in cleaned and len(tokens) > 1:
        return tokens[-1]
    return tokens[0]
