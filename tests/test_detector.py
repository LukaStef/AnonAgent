"""Detector tests against the real Presidio engine.

These load a spaCy model, so they are slower than the rest of the suite.
Run just the fast tests with: pytest -m "not slow"
"""

import pytest

from anonagent.privacy.detector import PiiDetector

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def detector():
    return PiiDetector()


def types_in(detector, text):
    return {entity.entity_type for entity in detector.detect(text)}


def test_detects_a_person_and_an_email(detector):
    found = types_in(detector, "James Bond can be reached at james@example.com.")
    assert "PERSON" in found
    assert "EMAIL_ADDRESS" in found


def test_detects_a_us_ssn(detector):
    assert "US_SSN" in types_in(detector, "Patient SSN is 432-56-7890.")


def test_an_ambiguous_number_is_still_masked(detector):
    """Mislabeling is acceptable; letting a value through is not.

    078-05-1120 reads as a phone number to Presidio rather than an SSN. The
    label on the placeholder is then wrong, but the digits are still replaced,
    which is the property that actually protects the user.
    """
    entities = detector.detect("Patient SSN is 078-05-1120.")
    assert [e.text for e in entities] == ["078-05-1120"]


def test_detects_an_iban(detector):
    assert "IBAN_CODE" in types_in(detector, "Transfer to RS35260005601001611379 today.")


def test_detects_a_custom_employee_id(detector):
    assert "EMPLOYEE_ID" in types_in(detector, "Filed by EMP-004217 last night.")


def test_detects_an_api_key(detector):
    text = "client = OpenAI(api_key='sk-proj-4Xq82LmZpR7vNw1KdTfA9BcE')"
    assert "API_KEY" in types_in(detector, text)


def test_money_is_ignored_unless_asked_for(detector):
    text = "Approve the transfer of 50,000 RSD."
    assert "MONEY_AMOUNT" not in types_in(detector, text)
    assert "MONEY_AMOUNT" in types_in(PiiDetector(detect_money=True), text)


def test_clean_text_yields_nothing(detector):
    assert detector.detect("Summarize the quarterly revenue trend.") == []


def test_detected_spans_never_overlap(detector):
    text = (
        "James Bond, SSN 078-05-1120, emailed james@example.com "
        "from 10.0.14.201 on behalf of EMP-004217."
    )
    entities = detector.detect(text)
    for earlier, later in zip(entities, entities[1:]):
        assert earlier.end <= later.start


def test_detected_text_matches_its_own_offsets(detector):
    text = "James Bond can be reached at james@example.com."
    for entity in detector.detect(text):
        assert text[entity.start : entity.end] == entity.text
