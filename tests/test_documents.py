"""Splitting documents, and masking them as one piece of work."""

import pytest
from helpers import StubDetector, ValueDetector, span

from anonagent.documents import DEFAULT_CHUNK_CHARS, split_into_chunks
from anonagent.privacy.detector import DetectedEntity
from anonagent.privacy.masker import Masker

PARAGRAPHS = "Sentence one here. Sentence two here.\n\n" * 400
LONG_SENTENCE = "word " * 4000
NO_BREAKS = "x" * 30_000


@pytest.mark.parametrize(
    "text",
    ["", "hello", PARAGRAPHS, LONG_SENTENCE, NO_BREAKS, "a\n\nb", "\n\n\n", "  "],
)
def test_chunks_always_rejoin_into_the_original(text):
    """The one invariant everything else depends on."""
    assert "".join(split_into_chunks(text, 8_000)) == text


@pytest.mark.parametrize("size", [1, 10, 100, 8_000])
def test_chunks_respect_the_limit_when_the_text_allows(size):
    chunks = split_into_chunks(PARAGRAPHS, size)
    # A chunk may exceed the limit only when a single atom does.
    assert all(len(c) <= size or " " not in c.strip() or size < 40 for c in chunks)


def test_a_short_text_is_one_chunk():
    assert split_into_chunks("Patient Maria Whitfield.", 8_000) == [
        "Patient Maria Whitfield."
    ]


def test_an_empty_text_is_no_chunks():
    assert split_into_chunks("", 8_000) == []


def test_cuts_prefer_paragraph_breaks():
    text = "A" * 100 + "\n\n" + "B" * 100
    assert split_into_chunks(text, 150) == ["A" * 100 + "\n\n", "B" * 100]


def test_cuts_fall_back_to_sentence_ends():
    text = "A" * 90 + ". " + "B" * 90 + "."
    chunks = split_into_chunks(text, 120)
    assert chunks[0].endswith(". ")
    assert "".join(chunks) == text


def test_an_impossible_limit_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        split_into_chunks("text", 0)


def document_masker(text, values):
    detector = StubDetector([span(text, value, kind) for value, kind in values])
    return Masker(detector=detector)


def test_one_person_keeps_one_placeholder_across_chunks():
    """The whole reason chunks share a vault."""
    filler = "Nothing sensitive here at all.\n\n" * 20
    text = "Maria Whitfield opened the file.\n\n" + filler + "Maria Whitfield closed it."

    masker = Masker(detector=ValueDetector([("Maria Whitfield", "PERSON")]))
    result = masker.mask_document(text, max_chars=200)

    assert result.masked_text.count("<PERSON_001>") == 2
    assert "Maria Whitfield" not in result.masked_text
    assert len(masker.vault) == 1


def test_a_later_chunk_is_covered_even_when_detection_fails_there():
    """The sweep carries what an earlier chunk taught the vault."""
    filler = "Nothing sensitive here at all.\n\n" * 20
    text = "Maria Whitfield opened the file.\n\n" + filler + "Maria Whitfield closed it."

    # Only ever reports the first occurrence, like NER losing the thread.
    masker = Masker(detector=StubDetector([span(text, "Maria Whitfield", "PERSON")]))
    result = masker.mask_document(text, max_chars=200)

    assert "Maria Whitfield" not in result.masked_text
    assert result.sealed == ["Maria Whitfield"]


def test_a_masked_document_restores_to_the_original():
    text = "Maria Whitfield called.\n\n" + ("Filler line.\n\n" * 30) + "Ask Maria Whitfield."
    masker = Masker(detector=ValueDetector([("Maria Whitfield", "PERSON")]))

    result = masker.mask_document(text, max_chars=200)
    assert masker.unmask(result.masked_text) == text


def test_offsets_are_absolute_not_chunk_relative():
    text = "Filler.\n\n" * 30 + "Maria Whitfield signed."
    detector = ValueDetector([("Maria Whitfield", "PERSON")])

    result = Masker(detector=detector).mask_document(text, max_chars=100)
    found = result.entities[0]
    assert text[found.start : found.end] == "Maria Whitfield"


def test_structure_outside_the_masked_values_is_untouched():
    text = "# Title\n\nMaria Whitfield wrote this.\n\n- bullet one\n- bullet two\n"
    detector = ValueDetector([("Maria Whitfield", "PERSON")])

    result = Masker(detector=detector).mask_document(text, max_chars=40)
    assert result.masked_text.startswith("# Title\n\n")
    assert result.masked_text.endswith("- bullet one\n- bullet two\n")


def test_a_document_with_nothing_sensitive_comes_back_unchanged():
    text = "Quarterly revenue rose.\n\n" * 50
    result = Masker(detector=StubDetector([])).mask_document(text, max_chars=200)
    assert result.masked_text == text
    assert result.entities == []


def test_the_default_chunk_size_is_in_the_fast_regime():
    """Measured: 10 KB takes 0.3 s, 500 KB takes 107 s. Stay small."""
    assert DEFAULT_CHUNK_CHARS <= 16_000
