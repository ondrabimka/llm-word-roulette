import numpy as np

from llm_word_roulette.metrics import jensen_shannon


def test_js_self_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert jensen_shannon(p, p) == 0.0


def test_js_disjoint_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert abs(jensen_shannon(p, q) - 1.0) < 1e-12
