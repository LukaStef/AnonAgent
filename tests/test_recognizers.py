"""The recognizers that do not depend on a statistical model.

Named-entity recognition is uneven by nature: it found "Slobodan Zivkovic"
and missed "Letar Pukovac" in the same document. These rules do not care how
familiar a value looks.
"""

import pytest

from anonagent.privacy.detector import PiiDetector
from anonagent.privacy.recognizers import _valid_jmbg

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def detector():
    return PiiDetector()


def found(detector, text):
    return {(e.entity_type, e.text) for e in detector.detect(text)}


def test_a_labelled_name_is_caught_however_unfamiliar(detector):
    """The regression that started this: NER knew one name and not the other."""
    text = "Full name: Slobodan Zivkovic\nFull name: Letar Pukovac"
    names = {e.text for e in detector.detect(text) if e.entity_type == "PERSON"}
    assert names == {"Slobodan Zivkovic", "Letar Pukovac"}


@pytest.mark.parametrize(
    "label", ["Full name", "Name", "Patient", "Client", "Account holder", "Employee"]
)
def test_the_labels_that_introduce_a_person(detector, label):
    entities = detector.detect(f"{label}: Letar Pukovac")
    assert ("PERSON", "Letar Pukovac") in {(e.entity_type, e.text) for e in entities}


def test_a_labelled_id_number_is_caught(detector):
    assert ("NATIONAL_ID", "4428374023740") in found(detector, "ID number: 4428374023740")


def test_an_unlabelled_national_id_is_still_caught(detector):
    assert ("NATIONAL_ID", "0412987654321") in found(
        detector, "The number 0412987654321 appeared twice."
    )


def test_a_valid_jmbg_outranks_the_phone_number_reading(detector):
    """0412987654321 used to be masked as a phone number, which read wrong."""
    entities = detector.detect("ID number: 0412987654321")
    assert [e.entity_type for e in entities] == ["NATIONAL_ID"]


def test_a_label_we_do_not_know_is_left_alone(detector):
    assert found(detector, "Transaction: wire transfer approved") == set()


def test_ordinary_prose_with_a_colon_is_not_a_field(detector):
    assert found(detector, "Note: please review this before Friday.") == set()


def test_labelled_fields_survive_a_realistic_record(detector):
    text = (
        "Full name: Letar Pukovac\n"
        "ID number: 4428374023740\n"
        "Email: letar@example.com\n"
        "Phone: +1 415 555 0182\n"
    )
    types = {e.entity_type for e in detector.detect(text)}
    assert {"PERSON", "NATIONAL_ID", "EMAIL_ADDRESS", "PHONE_NUMBER"} <= types


@pytest.mark.parametrize("number", ["0101990500003", "1503857100129", "0412987654321"])
def test_the_checksum_accepts_a_well_formed_jmbg(number):
    assert _valid_jmbg(number) is True


@pytest.mark.parametrize("number", ["4428374023740", "1234567890123"])
def test_a_failed_checksum_is_undecided_rather_than_cleared(number):
    """Failing the checksum does not make a number safe to publish."""
    assert _valid_jmbg(number) is None


def test_the_checksum_ignores_anything_not_thirteen_digits():
    assert _valid_jmbg("123") is None


BAD_IBAN = "DE99370400440532013000"
GOOD_IBAN = "DE89370400440532013000"


@pytest.fixture(scope="module")
def lenient():
    return PiiDetector(mask_malformed=True)


def test_by_default_only_a_valid_account_is_masked(detector):
    """The chosen trade: precision over coverage.

    An account number that fails its checksum is left in the clear, which is
    the cost of not masking anything that merely looks like one.
    """
    assert ("IBAN_CODE", GOOD_IBAN) in found(detector, f"transfer to IBAN {GOOD_IBAN}")
    assert found(detector, f"transfer to IBAN {BAD_IBAN}") == set()


def test_by_default_only_a_valid_identity_number_is_masked(detector):
    assert ("NATIONAL_ID", "0412987654321") in found(detector, "The number 0412987654321.")
    assert found(detector, "The number 4428374023740.") == set()


def test_the_flag_covers_malformed_identifiers_too(lenient):
    assert ("IBAN_CODE", BAD_IBAN) in found(lenient, f"transfer to IBAN {BAD_IBAN}")
    assert ("NATIONAL_ID", "4428374023740") in found(lenient, "The number 4428374023740.")


def test_a_valid_account_still_scores_higher_than_a_shape(lenient):
    valid = lenient.detect(f"transfer to IBAN {GOOD_IBAN}")[0]
    invalid = lenient.detect(f"transfer to IBAN {BAD_IBAN}")[0]
    assert valid.score > invalid.score
    assert valid.entity_type == invalid.entity_type == "IBAN_CODE"


def test_a_labelled_field_needs_no_checksum(detector):
    """Names have no checksum, and neither does a field that names itself."""
    assert ("NATIONAL_ID", "534532532534") in found(detector, "ID number: 534532532534")


@pytest.mark.parametrize(
    "innocent",
    [
        "Order SKU AB12CDEFGHIJK shipped today.",
        "Commit 4f9a2b7c1e8d3a5b6c7d8e9f0a1b2c3d4e5f6a7b landed.",
        "Ticket 9f3e1c22-7b4a-4d58-9e21-0f7a6b5c4d3e was closed.",
        "The ISO 27001 audit passed.",
        "Flight LH1234 departs at noon.",
    ],
)
def test_ordinary_identifiers_are_not_mistaken_for_accounts(lenient, innocent):
    """Even with the flag on, an ordinary code is not an account."""
    assert found(lenient, innocent) == set()


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("ID number: 343", "too short to be an identity number"),
        ("ID number: 1", "a single digit"),
        ("ID number: x", "not even a number"),
        ("Full name: A", "one letter is not a name"),
        ("Email: not-an-email", "no @"),
        ("Phone: banana", "no digits"),
    ],
)
def test_a_label_is_not_believed_over_an_implausible_value(detector, field, reason):
    """The label says what the field is for, not what someone typed into it."""
    assert found(detector, field) == set(), reason


@pytest.mark.parametrize(
    "field",
    [
        "ID number: 534532532534",
        "Full name: Letar Pukovac",
        "Email: letar@example.com",
        "Phone: +1 415 555 0182",
    ],
)
def test_a_plausible_labelled_value_is_still_masked(detector, field):
    assert found(detector, field) != set()


def test_the_label_word_itself_is_never_masked(detector):
    """NER read the word "Email" as a person, which masked the label.

    A field label is structure, not data. Masking it produces output that
    reads like nonsense while the value beside it survives.
    """
    entities = detector.detect("Email: letar@example.com")
    assert [e.text for e in entities] == ["letar@example.com"]


def test_a_person_label_does_not_swallow_the_word_before_the_colon(detector):
    entities = detector.detect("Patient: Letar Pukovac")
    assert [e.text for e in entities] == ["Letar Pukovac"]
