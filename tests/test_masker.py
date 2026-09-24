"""Masker tests with a stubbed detector.

The detector is stubbed so these stay fast and deterministic: they test the
substitution arithmetic, not whether spaCy recognizes a given name.
"""

import pytest

from anonagent.privacy.detector import DetectedEntity
from anonagent.privacy.masker import Masker
from helpers import StubDetector, span


def test_round_trip_restores_the_original_exactly():
    text = "Nikola Stefanovic emailed nikola@example.com about the audit."
    detector = StubDetector(
        [span(text, "Nikola Stefanovic", "PERSON"), span(text, "nikola@example.com", "EMAIL_ADDRESS")]
    )
    masker = Masker(detector=detector)

    result = masker.mask(text)
    assert masker.unmask(result.masked_text) == text


def test_masked_text_contains_no_original_value():
    text = "Nikola Stefanovic emailed nikola@example.com about the audit."
    detector = StubDetector(
        [span(text, "Nikola Stefanovic", "PERSON"), span(text, "nikola@example.com", "EMAIL_ADDRESS")]
    )
    result = Masker(detector=detector).mask(text)

    assert "Nikola Stefanovic" not in result.masked_text
    assert "nikola@example.com" not in result.masked_text
    assert "<PERSON_001>" in result.masked_text
    assert "about the audit." in result.masked_text


def test_repeated_value_collapses_to_one_placeholder():
    text = "Nikola signed it, then Nikola countersigned it."
    first = span(text, "Nikola", "PERSON")
    second = DetectedEntity("PERSON", text.rindex("Nikola"), text.rindex("Nikola") + 6, 0.9, "Nikola")
    masker = Masker(detector=StubDetector([first, second]))

    result = masker.mask(text)
    assert result.masked_text.count("<PERSON_001>") == 2
    assert len(masker.vault) == 1


def test_masking_survives_the_model_rewriting_the_sentence():
    """The model answers in its own words; only the placeholders come back."""
    text = "Nikola Stefanovic requested the transfer."
    masker = Masker(detector=StubDetector([span(text, "Nikola Stefanovic", "PERSON")]))
    masker.mask(text)

    model_reply = "Yes, PERSON_001 is authorized. Notify **<PERSON_001>** by Friday."
    assert masker.unmask(model_reply) == (
        "Yes, Nikola Stefanovic is authorized. Notify **Nikola Stefanovic** by Friday."
    )


def test_text_with_nothing_sensitive_passes_through_untouched():
    text = "Summarize the quarterly revenue trend."
    result = Masker(detector=StubDetector([])).mask(text)
    assert result.masked_text == text
    assert result.entities == []


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_is_handled(text):
    result = Masker(detector=StubDetector([])).mask(text)
    assert result.masked_text == text


def test_entity_counts_describe_without_disclosing():
    text = "Nikola Stefanovic emailed nikola@example.com about the audit."
    detector = StubDetector(
        [span(text, "Nikola Stefanovic", "PERSON"), span(text, "nikola@example.com", "EMAIL_ADDRESS")]
    )
    counts = Masker(detector=detector).mask(text).entity_counts

    assert counts == {"PERSON": 1, "EMAIL_ADDRESS": 1}
    assert "Nikola" not in str(counts)


def test_an_empty_vault_passed_in_is_actually_used():
    """Regression: PseudonymVault defines __len__, so an empty one is falsy.

    Building the default with `vault or PseudonymVault()` discarded the
    caller's vault, and every option set on it went silently missing.
    """
    from anonagent.privacy.vault import PseudonymVault

    vault = PseudonymVault(match="exact")
    masker = Masker(detector=StubDetector([]), vault=vault)
    assert masker.vault is vault
    assert masker.vault.match == "exact"


def test_exact_mode_reaches_the_vault_through_the_masker():
    text = "Mail nikola@example.com and Nikola@Example.COM"
    from anonagent.privacy.vault import PseudonymVault

    detector = StubDetector(
        [
            span(text, "nikola@example.com", "EMAIL_ADDRESS"),
            span(text, "Nikola@Example.COM", "EMAIL_ADDRESS"),
        ]
    )
    masker = Masker(detector=detector, vault=PseudonymVault(match="exact"))
    result = masker.mask(text)

    assert "<EMAIL_ADDRESS_002>" in result.masked_text
    assert masker.unmask(result.masked_text) == text


def test_adjacent_spans_do_not_corrupt_each_other():
    text = "AB"
    detector = StubDetector(
        [
            DetectedEntity("PERSON", 0, 1, 0.9, "A"),
            DetectedEntity("PERSON", 1, 2, 0.9, "B"),
        ]
    )
    masker = Masker(detector=detector)
    result = masker.mask(text)

    assert result.masked_text == "<PERSON_001><PERSON_002>"
    assert masker.unmask(result.masked_text) == text
