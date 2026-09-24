"""Masking and unmasking: the round trip around the cloud model."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from anonagent.documents import DEFAULT_CHUNK_CHARS, split_into_chunks
from anonagent.privacy.detector import DetectedEntity, PiiDetector
from anonagent.privacy.vault import Pseudonym, PseudonymVault
from anonagent.review import Decision, mask_everything


@dataclass(frozen=True)
class MaskingResult:
    """What one masking pass produced."""

    original_text: str
    masked_text: str
    entities: list[DetectedEntity] = field(default_factory=list)
    pseudonyms: list[Pseudonym] = field(default_factory=list)
    sealed: list[str] = field(default_factory=list)
    declined: list[DetectedEntity] = field(default_factory=list)

    @property
    def entity_counts(self) -> dict[str, int]:
        """How many spans of each type were masked.

        Safe to log and to show on screen: it describes the secrets without
        containing any of them.
        """
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
        return counts


class Masker:
    """Turns sensitive text into text that is safe to send out, and back."""

    def __init__(
        self,
        detector: PiiDetector | None = None,
        vault: PseudonymVault | None = None,
        reviewer: Callable[[list[DetectedEntity]], Decision] | None = None,
    ) -> None:
        # `is None`, not `or`: PseudonymVault defines __len__, so an empty one
        # is falsy and `vault or PseudonymVault()` would silently discard it.
        self.detector = PiiDetector() if detector is None else detector
        self.vault = PseudonymVault() if vault is None else vault
        self.reviewer = mask_everything if reviewer is None else reviewer

    def mask(self, text: str) -> MaskingResult:
        # The reviewer can drop uncertain detections, and a dropped value must
        # never reach the vault: the sweep would otherwise seal a value the
        # user explicitly chose to leave in the clear.
        entities, declined = self.reviewer(self.detector.detect(text))

        # Mint front to back, so PERSON_001 is the person mentioned first.
        pseudonyms: list[Pseudonym] = [
            self.vault.pseudonymize(entity.text, entity.entity_type) for entity in entities
        ]

        # Substitute back to front: an edit near the end cannot shift the
        # offsets of a span that starts earlier, so every span stays valid.
        masked = text
        for entity, pseudonym in zip(reversed(entities), reversed(pseudonyms)):
            masked = masked[: entity.start] + pseudonym.placeholder + masked[entity.end :]

        masked, sealed = self._sweep(masked)

        return MaskingResult(
            original_text=text,
            masked_text=masked,
            entities=entities,
            pseudonyms=pseudonyms,
            sealed=sealed,
            declined=declined,
        )

    def mask_document(
        self, text: str, max_chars: int = DEFAULT_CHUNK_CHARS
    ) -> MaskingResult:
        """Mask a whole document, one chunk at a time, into one result.

        The chunks share this masker's vault, so a person named in the first
        paragraph and again in the last is one placeholder throughout. Order
        helps too: once a chunk teaches the vault a value, the sweep seals
        that value in every chunk that follows, even where the detector would
        have missed it.
        """
        chunks = split_into_chunks(text, max_chars)
        masked_parts: list[str] = []
        entities: list[DetectedEntity] = []
        pseudonyms: list[Pseudonym] = []
        sealed: list[str] = []
        declined: list[DetectedEntity] = []

        offset = 0
        for chunk in chunks:
            result = self.mask(chunk)
            masked_parts.append(result.masked_text)
            # Chunk-relative offsets mean nothing to a caller holding the
            # whole document, so shift them as the chunks are consumed.
            entities.extend(_shifted(result.entities, offset))
            declined.extend(_shifted(result.declined, offset))
            pseudonyms.extend(result.pseudonyms)
            sealed.extend(result.sealed)
            offset += len(chunk)

        return MaskingResult(
            original_text=text,
            masked_text="".join(masked_parts),
            entities=entities,
            pseudonyms=pseudonyms,
            sealed=sealed,
            declined=declined,
        )

    def unmask(self, text: str) -> str:
        return self.vault.restore(text)

    def redact(self, text: str) -> str:
        """Replace every value the vault knows about with its placeholder.

        For text that did not come from the user at all -- an error message
        from a provider SDK, say. Third-party strings echo back whatever they
        please, so anything on its way to a screen or a log goes through here
        first.
        """
        return self._sweep(text)[0]

    def leaks(self, text: str) -> list[str]:
        """Known private values still present in ``text``.

        A non-empty result means something is about to leave the machine that
        should not. Callers are expected to treat it as fatal.
        """
        found: list[str] = []
        for pseudonym in self.vault.entries():
            for variant in self.vault.variants(pseudonym):
                if _literal_pattern(variant).search(text):
                    found.append(variant)
        return found

    def _sweep(self, masked: str) -> tuple[str, list[str]]:
        """Replace any known value the detector missed on a later mention.

        The detector works span by span and can recognize a name in one
        sentence but miss it in the next. Once the vault knows a value is
        private, every later occurrence of it is private too, regardless of
        what the model that found it thinks the second time.

        Matching is case-insensitive even in "exact" mode: bending the
        byte-exact promise for text that would otherwise leak is the right way
        round of that trade.
        """
        sealed: list[str] = []
        for pseudonym in self.vault.entries():
            for variant in self.vault.variants(pseudonym):
                masked, hits = _literal_pattern(variant).subn(pseudonym.placeholder, masked)
                if hits:
                    sealed.append(variant)
        return masked, sealed


def _shifted(entities: list[DetectedEntity], offset: int) -> list[DetectedEntity]:
    return [replace(e, start=e.start + offset, end=e.end + offset) for e in entities]


def _literal_pattern(value: str) -> re.Pattern[str]:
    """Match ``value`` literally, without matching inside a longer word.

    Word boundaries only go where they mean something: anchoring \\b against
    a value that starts or ends with punctuation would never match.
    """
    prefix = r"\b" if value[:1].isalnum() else ""
    suffix = r"\b" if value[-1:].isalnum() else ""
    return re.compile(prefix + re.escape(value) + suffix, re.IGNORECASE)
