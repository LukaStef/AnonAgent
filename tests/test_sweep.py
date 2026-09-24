"""The safety sweep: values the detector found once and missed later.

NER is span-by-span and inconsistent. Once the vault knows a value is
private, every later occurrence of it is private too.
"""

from helpers import StubDetector, span

from anonagent.privacy.masker import Masker
from anonagent.privacy.vault import PseudonymVault


def test_a_second_mention_the_detector_missed_is_still_masked():
    text = "James Bond filed it. Ask James Bond for the date."
    detector = StubDetector([span(text, "James Bond", "PERSON")])

    result = Masker(detector=detector).mask(text)
    assert "James Bond" not in result.masked_text
    assert result.masked_text.count("<PERSON_001>") == 2


def test_the_sweep_reports_what_it_caught():
    text = "James Bond filed it. Ask James Bond for the date."
    detector = StubDetector([span(text, "James Bond", "PERSON")])

    assert Masker(detector=detector).mask(text).sealed == ["James Bond"]


def test_a_differently_cased_second_mention_is_caught():
    text = "James Bond filed it. Ask JAMES BOND for the date."
    detector = StubDetector([span(text, "James Bond", "PERSON")])

    result = Masker(detector=detector).mask(text)
    assert "JAMES BOND" not in result.masked_text


def test_the_sweep_runs_in_exact_mode_too():
    """Bending byte-exactness beats leaking. Documented, deliberate."""
    text = "Mail james@example.com. Then mail JAMES@EXAMPLE.COM."
    detector = StubDetector([span(text, "james@example.com", "EMAIL_ADDRESS")])
    masker = Masker(detector=detector, vault=PseudonymVault(match="exact"))

    result = masker.mask(text)
    assert "JAMES@EXAMPLE.COM" not in result.masked_text


def test_a_short_value_does_not_get_masked_inside_a_longer_word():
    text = "Al signed it. Also, the deadline moved."
    detector = StubDetector([span(text, "Al", "PERSON")])

    result = Masker(detector=detector).mask(text)
    assert "Also, the deadline moved." in result.masked_text
    assert result.masked_text.startswith("<PERSON_001> signed it.")


def test_nothing_is_swept_when_the_detector_caught_everything():
    text = "James Bond filed it."
    detector = StubDetector([span(text, "James Bond", "PERSON")])

    assert Masker(detector=detector).mask(text).sealed == []


def test_leaks_reports_a_value_that_survived():
    masker = Masker(detector=StubDetector([]))
    masker.vault.pseudonymize("James Bond", "PERSON")

    assert masker.leaks("Ask James Bond.") == ["James Bond"]
    assert masker.leaks("Ask <PERSON_001>.") == []


def test_a_swept_text_reports_no_leaks():
    text = "James Bond filed it. Ask James Bond for the date."
    detector = StubDetector([span(text, "James Bond", "PERSON")])
    masker = Masker(detector=detector)

    result = masker.mask(text)
    assert masker.leaks(result.masked_text) == []
