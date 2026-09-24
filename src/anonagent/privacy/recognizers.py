"""Custom Presidio recognizers for formats its defaults do not cover.

Presidio ships recognizers for names, emails, phone numbers, credit cards,
IBANs and government IDs. These add the organization-specific formats that
make the guard useful in a real deployment.
"""

from __future__ import annotations

import re

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

EMPLOYEE_ID = "EMPLOYEE_ID"
API_KEY = "API_KEY"
MONEY_AMOUNT = "MONEY_AMOUNT"
NATIONAL_ID = "NATIONAL_ID"


def employee_id_recognizer() -> PatternRecognizer:
    """Internal staff identifiers, e.g. ``EMP-004217``."""
    return PatternRecognizer(
        supported_entity=EMPLOYEE_ID,
        patterns=[Pattern(name="employee_id", regex=r"\bEMP-\d{6}\b", score=0.9)],
    )


def api_key_recognizer() -> PatternRecognizer:
    """Credentials that people paste into prompts without thinking."""
    return PatternRecognizer(
        supported_entity=API_KEY,
        patterns=[
            Pattern(name="openai_key", regex=r"\bsk-[A-Za-z0-9_\-]{20,}\b", score=0.95),
            Pattern(name="github_token", regex=r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", score=0.95),
            Pattern(name="aws_access_key", regex=r"\bAKIA[0-9A-Z]{16}\b", score=0.95),
            Pattern(name="bearer_token", regex=r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", score=0.85),
        ],
    )


def money_amount_recognizer() -> PatternRecognizer:
    """Currency amounts.

    Off by default, and that is a design decision rather than an oversight:
    masking the numbers is what stops the model from answering questions like
    "is this balance enough for the transfer". Turn it on when the amounts
    themselves are the secret.
    """
    return PatternRecognizer(
        supported_entity=MONEY_AMOUNT,
        patterns=[
            Pattern(
                name="symbol_first",
                regex=r"(?:[$€£]|\b(?:USD|EUR|GBP|RSD)\b)\s?\d[\d.,]*",
                score=0.7,
            ),
            Pattern(
                name="currency_last",
                regex=r"\b\d[\d.,]*\s?(?:[$€£]|USD|EUR|GBP|RSD|dollars|euros|pounds)\b",
                score=0.7,
            ),
        ],
    )


class NationalIdRecognizer(PatternRecognizer):
    """Thirteen-digit national identity numbers, JMBG among them.

    With ``mask_malformed`` a number that fails the JMBG checksum is still
    treated as sensitive, on the grounds that a mistyped or foreign identity
    number is nobody's business either. Without it, only numbers that check
    out are masked, and a mistyped one travels in the clear.
    """

    def __init__(self, *, mask_malformed: bool = False) -> None:
        super().__init__(
            supported_entity=NATIONAL_ID,
            patterns=[Pattern(name="thirteen_digits", regex=r"\b\d{13}\b", score=0.45)],
            context=["jmbg", "id number", "identity", "national", "personal number"],
        )
        self.mask_malformed = mask_malformed

    def validate_result(self, pattern_text: str) -> bool | None:
        valid = _valid_jmbg(pattern_text)
        if valid is True:
            return True
        # None keeps the pattern score, so the value is still masked.
        # False discards it, so only a checksummed number counts.
        return None if self.mask_malformed else False


def national_id_recognizer(*, mask_malformed: bool = False) -> PatternRecognizer:
    return NationalIdRecognizer(mask_malformed=mask_malformed)


def _valid_jmbg(text: str) -> bool | None:
    """The JMBG checksum. Returns None for "cannot tell", never False.

    A number that fails the checksum is not thereby innocent -- it may be a
    foreign identifier, or simply mistyped -- so a failure leaves the pattern
    score standing rather than clearing the value for publication.
    """
    digits = [int(character) for character in text if character.isdigit()]
    if len(digits) != 13:
        return None
    weights = (7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
    total = sum(weight * digit for weight, digit in zip(weights, digits))
    remainder = 11 - (total % 11)
    expected = remainder if remainder < 10 else 0
    return True if expected == digits[12] else None


def iban_like_recognizer() -> PatternRecognizer:
    """Account numbers shaped like an IBAN, including ones that do not check out.

    Presidio validates the IBAN checksum and discards anything that fails it,
    which is right for deciding "is this a real IBAN" and wrong for deciding
    "is this somebody's account number". A mistyped or foreign account is
    still private, and a document about suspected fraud is exactly where
    malformed identifiers turn up.

    Scored below a validated IBAN, so a real one keeps the better label.
    """
    return PatternRecognizer(
        supported_entity="IBAN_CODE",
        patterns=[
            Pattern(
                name="iban_shaped",
                regex=r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
                score=0.5,
            )
        ],
        context=["iban", "account", "transfer", "wire"],
    )


#: Labels that introduce a value, and what that value is. Documents in this
#: shape -- forms, records, transaction reports -- say what each field holds,
#: which is a far stronger signal than guessing from the value itself.
FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "PERSON": (
        "full name", "name", "patient", "client", "customer",
        "account holder", "holder", "beneficiary", "applicant", "employee",
    ),
    NATIONAL_ID: (
        "id number", "id no", "identity number", "jmbg", "national id",
        "personal number", "ssn", "social security number", "passport",
    ),
    "LOCATION": ("address", "residence", "street"),
    "EMAIL_ADDRESS": ("email", "e-mail", "mail"),
    "PHONE_NUMBER": ("phone", "telephone", "tel", "mobile", "contact number"),
    "IBAN_CODE": ("iban", "account number", "account"),
    "CREDIT_CARD": ("card number", "card"),
}


#: What a labelled value has to look like before the label is believed.
#: A form that says "ID number:" is a strong signal, but it is a signal about
#: the field, not a guarantee about what someone typed into it. These are
#: plausibility floors, not checksums: enough to reject "343" and "banana"
#: without demanding that every identifier be well formed.
PLAUSIBLE: dict[str, tuple[int, int]] = {
    # entity type -> (minimum characters, minimum digits)
    NATIONAL_ID: (5, 4),
    "PHONE_NUMBER": (6, 5),
    "IBAN_CODE": (10, 4),
    "CREDIT_CARD": (12, 12),
    "PERSON": (2, 0),
    "LOCATION": (3, 0),
    "EMAIL_ADDRESS": (5, 0),
}


def _plausible(entity_type: str, value: str) -> bool:
    if entity_type == "EMAIL_ADDRESS" and "@" not in value:
        return False
    minimum_length, minimum_digits = PLAUSIBLE.get(entity_type, (1, 0))
    if len(value) < minimum_length:
        return False
    return sum(character.isdigit() for character in value) >= minimum_digits


class LabelledFieldRecognizer(EntityRecognizer):
    """Reads "Label: value" lines and trusts the label.

    Statistical name recognition is uneven: it will find one name and miss
    the next one in the same document, depending on how much each resembles
    what the model was trained on. A line reading "Full name:" does not care
    how familiar the name is, and records, forms and reports are written in
    exactly that shape.
    """

    _PATTERN = re.compile(
        r"^[ \t]*(?P<label>[A-Za-z][A-Za-z ./-]{1,28}?)[ \t]*:[ \t]*(?P<value>\S.*?)[ \t]*$",
        re.MULTILINE,
    )

    def __init__(self, score: float = 0.8) -> None:
        super().__init__(supported_entities=sorted(FIELD_LABELS), name="LabelledField")
        self.score = score
        self._by_label = {
            label: entity_type
            for entity_type, labels in FIELD_LABELS.items()
            for label in labels
        }

    def load(self) -> None:  # required by the base class, nothing to load
        return None

    def analyze(self, text, entities, nlp_artifacts=None) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        for match in self._PATTERN.finditer(text):
            entity_type = self._by_label.get(match.group("label").strip().lower())
            if entity_type is None or entity_type not in entities:
                continue
            if not _plausible(entity_type, match.group("value")):
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=match.start("value"),
                    end=match.end("value"),
                    score=self.score,
                    analysis_explanation=None,
                )
            )
        return results


def label_spans(text: str) -> list[tuple[int, int]]:
    """Where the labels themselves sit.

    The word "Email" in "Email: ..." is structure, not data, and a statistical
    model that reads it as somebody's name would otherwise have the label
    masked while the address beside it stayed put.
    """
    spans: list[tuple[int, int]] = []
    known = {label for labels in FIELD_LABELS.values() for label in labels}
    for match in LabelledFieldRecognizer._PATTERN.finditer(text):
        if match.group("label").strip().lower() in known:
            spans.append((match.start("label"), match.end("label")))
    return spans


def build_custom_recognizers(
    *, detect_money: bool = False, mask_malformed: bool = False
) -> list[EntityRecognizer]:
    """The custom recognizers to register with the analyzer.

    ``mask_malformed`` decides what happens to an identifier that fails its
    own checksum. Off, only identifiers that check out are masked. On, the
    shape alone is enough, because a mistyped account number still belongs to
    somebody.
    """
    recognizers: list[EntityRecognizer] = [
        employee_id_recognizer(),
        api_key_recognizer(),
        national_id_recognizer(mask_malformed=mask_malformed),
        LabelledFieldRecognizer(),
    ]
    if mask_malformed:
        recognizers.append(iban_like_recognizer())
    if detect_money:
        recognizers.append(money_amount_recognizer())
    return recognizers
