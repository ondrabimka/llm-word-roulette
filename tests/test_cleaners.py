from llm_word_roulette.cleaners import shakespeare_dialogue


def test_shakespeare_cleaner_removes_speaker_labels():
    text = "First Citizen:\nBefore we proceed any further.\nSecond Citizen:\nSpeak, speak.\n"
    cleaned, stats = shakespeare_dialogue(text)
    assert "First Citizen:" not in cleaned
    assert "Second Citizen:" not in cleaned
    assert "Before we proceed any further." in cleaned
    assert stats["speaker_lines_removed"] == 2
