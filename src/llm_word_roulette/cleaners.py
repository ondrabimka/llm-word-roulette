from __future__ import annotations

import re


SPEAKER_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z '\-]*:\s*$")


def shakespeare_dialogue(text: str) -> tuple[str, dict[str, int]]:
    removed = 0
    kept_lines: list[str] = []
    for line in text.splitlines():
        if SPEAKER_LINE_RE.match(line):
            removed += 1
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip() + "\n", {"speaker_lines_removed": removed}


def generic_plaintext(text: str) -> tuple[str, dict[str, int]]:
    start_markers = ["*** START OF", "***START OF"]
    end_markers = ["*** END OF", "***END OF"]
    lines = text.splitlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if any(marker in line for marker in start_markers):
            start_idx = i + 1
            break

    end_idx = len(lines)
    for i, line in enumerate(lines):
        if any(marker in line for marker in end_markers):
            end_idx = i
            break

    cleaned = "\n".join(lines[start_idx:end_idx])
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned, {"gutenberg_trimmed": int(start_idx > 0 or end_idx < len(lines))}


CLEANERS = {
    "shakespeare_dialogue": shakespeare_dialogue,
    "generic_plaintext": generic_plaintext,
}


def run_cleaner(name: str, text: str) -> tuple[str, dict[str, int]]:
    if name not in CLEANERS:
        raise KeyError(f"Unknown cleaner '{name}'. Available: {', '.join(sorted(CLEANERS))}")
    return CLEANERS[name](text)
