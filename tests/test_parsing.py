from llm_word_roulette.parsing import parse_chat_word


def test_parse_chat_word_cases():
    assert parse_chat_word("Love.") == "love"
    assert parse_chat_word(" 'Tis") == "tis"
    assert parse_chat_word("The word is: crown") == "crown"
    assert parse_chat_word("I can't choose randomly, but...") == "i"
    assert parse_chat_word("") is None
