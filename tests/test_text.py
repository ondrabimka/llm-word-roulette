from llm_word_roulette.text import tokenize_words


def test_tokenize_unicode_words():
    words = tokenize_words("příliš žluťoučký kůň")
    assert words == ["příliš", "žluťoučký", "kůň"]


def test_tokenize_apostrophes():
    words = tokenize_words("o'er and 'Tis")
    assert "o'er" in words
    assert "tis" in words
