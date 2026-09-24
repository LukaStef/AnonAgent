"""PII detection built on Microsoft Presidio."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from presidio_analyzer import AnalyzerEngine

from anonagent.privacy.recognizers import (
    API_KEY,
    EMPLOYEE_ID,
    MONEY_AMOUNT,
    NATIONAL_ID,
    build_custom_recognizers,
    label_spans,
)

#: Entities masked unless the caller says otherwise. DATE_TIME and URL are
#: left out on purpose: they are rarely the secret, and masking them tends to
#: strip the context the model needs to reason at all.
DEFAULT_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "LOCATION",
    EMPLOYEE_ID,
    API_KEY,
    NATIONAL_ID,
)


@dataclass(frozen=True)
class DetectedEntity:
    """One span of text the detector considers sensitive."""

    entity_type: str
    start: int
    end: int
    score: float
    text: str


class PiiDetector:
    """Finds sensitive spans in a piece of text.

    The Presidio engine loads a spaCy model on first use, which takes a few
    seconds, so construction stays cheap and the engine is built lazily.
    """

    def __init__(
        self,
        *,
        entities: tuple[str, ...] | None = None,
        language: str = "en",
        score_threshold: float = 0.4,
        detect_money: bool = False,
        mask_malformed: bool = False,
    ) -> None:
        self.language = language
        self.score_threshold = score_threshold
        self.detect_money = detect_money
        self.mask_malformed = mask_malformed
        self.entities = tuple(entities) if entities is not None else DEFAULT_ENTITIES
        if detect_money and MONEY_AMOUNT not in self.entities:
            self.entities += (MONEY_AMOUNT,)

    @cached_property
    def engine(self) -> AnalyzerEngine:
        engine = AnalyzerEngine(default_score_threshold=self.score_threshold)
        for recognizer in build_custom_recognizers(
            detect_money=self.detect_money, mask_malformed=self.mask_malformed
        ):
            engine.registry.add_recognizer(recognizer)
        return engine

    def detect(self, text: str) -> list[DetectedEntity]:
        """Return non-overlapping sensitive spans, ordered by position."""
        if not text.strip():
            return []

        results = self.engine.analyze(
            text=text,
            entities=list(self.entities),
            language=self.language,
        )
        found = [
            DetectedEntity(
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
                text=text[result.start : result.end],
            )
            for result in results
            if result.score >= self.score_threshold
        ]
        return _resolve_overlaps(_drop_labels(found, label_spans(text)))


def _drop_labels(
    entities: list[DetectedEntity], labels: list[tuple[int, int]]
) -> list[DetectedEntity]:
    """Discard anything found inside a field label rather than its value."""
    return [
        entity
        for entity in entities
        if not any(start <= entity.start and entity.end <= end for start, end in labels)
    ]


def _resolve_overlaps(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Drop spans that overlap a stronger one.

    Two recognizers often claim the same characters -- a phone number inside
    what the NER model read as an ID, say. Replacing overlapping spans would
    corrupt the text, so the most confident (then longest) span wins.
    """
    ranked = sorted(entities, key=lambda e: (-e.score, -(e.end - e.start), e.start))
    kept: list[DetectedEntity] = []
    for candidate in ranked:
        if any(candidate.start < k.end and k.start < candidate.end for k in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda e: e.start)
