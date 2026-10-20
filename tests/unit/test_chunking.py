import random
import re

import pytest
from pydantic import ValidationError

from retrieval_labs.documents import Contract
from retrieval_labs.ingestion.chunking import ChunkingConfig, chunk_contract

SMALL = ChunkingConfig(window_words=5, overlap_words=2)


def words_text(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def contract(text: str) -> Contract:
    return Contract.from_text("c1", "t", text)


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_config_rejects_overlap_not_smaller_than_window() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(window_words=5, overlap_words=5)


def test_default_config() -> None:
    config = ChunkingConfig()
    assert (config.window_words, config.overlap_words, config.stride) == (150, 50, 100)
    assert config.label == "words150-overlap50"


def test_chunk_text_is_exact_slice_of_original() -> None:
    text = "\n  Section 1.\tTerm of the  Agreement.\n\nSection 2. Law of New York.  "
    c = contract(text)
    for chunk in chunk_contract(c, SMALL):
        assert c.text[chunk.start : chunk.end] == chunk.text
        assert chunk.text == chunk.text.strip()


def test_windows_and_overlap() -> None:
    chunks = chunk_contract(contract(words_text(12)), SMALL)
    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]
    assert [c.chunk_id for c in chunks] == ["c1#0000", "c1#0001", "c1#0002", "c1#0003"]
    assert {c.chunking for c in chunks} == {"words5-overlap2"}


def test_no_redundant_trailing_chunk() -> None:
    # 8 words: the second window (w3..w7) already reaches the end.
    chunks = chunk_contract(contract(words_text(8)), SMALL)
    assert [c.text for c in chunks] == ["w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7"]


def test_short_and_whitespace_only_texts() -> None:
    assert [c.text for c in chunk_contract(contract("  one  "), SMALL)] == ["one"]
    assert chunk_contract(contract(" \n\t "), SMALL) == []


def test_every_word_is_covered() -> None:
    text = words_text(37)
    chunks = chunk_contract(contract(text), SMALL)
    for start, end in word_spans(text):
        assert any(c.start <= start and end <= c.end for c in chunks)


@pytest.mark.parametrize("n_words", [1, 4, 5, 6, 13, 50, 101])
def test_runs_up_to_overlap_plus_one_words_fit_in_one_chunk(n_words: int) -> None:
    text = words_text(n_words)
    spans = word_spans(text)
    chunks = chunk_contract(contract(text), SMALL)
    max_len = SMALL.overlap_words + 1
    for first in range(len(spans)):
        for length in range(1, max_len + 1):
            last = first + length - 1
            if last >= len(spans):
                break
            start, end = spans[first][0], spans[last][1]
            assert any(c.start <= start and end <= c.end for c in chunks), (first, length)


def test_longer_runs_are_not_guaranteed() -> None:
    """Documents the limit: a run of overlap + 2 words can straddle two chunks."""
    text = words_text(12)
    spans = word_spans(text)
    chunks = chunk_contract(contract(text), SMALL)
    start, end = spans[2][0], spans[5][1]  # w2..w5: 4 words = overlap + 2
    assert not any(c.start <= start and end <= c.end for c in chunks)


def test_random_texts_keep_offsets() -> None:
    rng = random.Random(0)
    alphabet = ["a", "b", " ", "\n", "\t", " ", ".", "é"]
    for _ in range(50):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 300)))
        if not text.strip():
            continue
        c = contract(text)
        for chunk in chunk_contract(c, SMALL):
            assert c.text[chunk.start : chunk.end] == chunk.text
