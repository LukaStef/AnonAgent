"""Overlap resolution is a pure function, so it needs no spaCy model."""

from anonagent.privacy.detector import DetectedEntity, _resolve_overlaps


def entity(entity_type, start, end, score):
    return DetectedEntity(entity_type, start, end, score, "x" * (end - start))


def test_the_stronger_of_two_overlapping_spans_wins():
    weak = entity("PERSON", 0, 10, 0.5)
    strong = entity("US_SSN", 4, 14, 0.9)
    assert _resolve_overlaps([weak, strong]) == [strong]


def test_the_longer_span_wins_when_scores_tie():
    short = entity("PERSON", 0, 5, 0.8)
    long = entity("PERSON", 0, 12, 0.8)
    assert _resolve_overlaps([short, long]) == [long]


def test_touching_but_not_overlapping_spans_are_both_kept():
    first = entity("PERSON", 0, 5, 0.9)
    second = entity("PERSON", 5, 10, 0.9)
    assert _resolve_overlaps([first, second]) == [first, second]


def test_survivors_come_back_in_document_order():
    late = entity("PERSON", 40, 50, 0.9)
    early = entity("EMAIL_ADDRESS", 0, 10, 0.6)
    assert [e.start for e in _resolve_overlaps([late, early])] == [0, 40]


def test_a_span_contained_in_a_stronger_one_is_dropped():
    outer = entity("IBAN_CODE", 0, 22, 0.9)
    inner = entity("PHONE_NUMBER", 5, 12, 0.4)
    assert _resolve_overlaps([outer, inner]) == [outer]


def test_empty_input():
    assert _resolve_overlaps([]) == []
