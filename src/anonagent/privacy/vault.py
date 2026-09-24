"""Reversible pseudonymization vault.

This is the only component that can turn ``<PERSON_001>`` back into a real
name. It lives in process memory for the lifetime of a session and never
touches the network.

Note that pseudonymization is deliberately *reversible*: a one-way hash would
make step 4 of the pipeline (restoring the real values in the model's answer)
impossible. Secrecy comes from the mapping never leaving this machine, not
from the placeholder being hard to invert.
"""

from __future__ import annotations

import hmac
import re
import secrets
from dataclasses import dataclass
from hashlib import sha256

PLACEHOLDER_TEMPLATE = "<{entity_type}_{index:03d}>"

#: Anything shaped like a placeholder: an entity name, a separator the model
#: may have changed, and a number whose zero padding it may have dropped.
#: The angle brackets are ours, so they are consumed. Any other punctuation
#: around the token belongs to the model's own prose and is left in place.
_PLACEHOLDER_SHAPE = re.compile(
    r"<?\b(?P<type>[A-Za-z][A-Za-z0-9_]*?)[_ \-](?P<index>\d{1,4})\b>?"
)


@dataclass(frozen=True)
class Pseudonym:
    """A single original value and the placeholder that stands in for it."""

    placeholder: str
    entity_type: str
    original: str
    fingerprint: str

    @property
    def bare(self) -> str:
        """The placeholder without its angle brackets."""
        return self.placeholder.strip("<>")


class PseudonymVault:
    """Two-way mapping between real values and their placeholders.

    The same value always receives the same placeholder within a vault, so a
    person mentioned three times in a document stays one person to the model.

    ``match`` decides when two mentions count as the same value:

    ``"normalized"``
        Case and repeated whitespace are ignored, so two spellings of one
        value that differ only in capitalisation or spacing share a single
        placeholder. The model sees one person instead of two, at the cost
        of restoration being *canonical* rather than byte-exact: every
        mention comes back spelled the way the first one was.
    ``"exact"``
        Only identical spellings share a placeholder, so restoration returns
        the text unchanged, character for character. Use it when the input
        must survive untouched -- source code, contracts, medical records.

    Pass a stable ``secret`` to make fingerprints comparable across sessions:
    two runs then agree on which values are the same value, without either run
    storing the value itself.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        secret: str | None = None,
        match: str = "normalized",
    ) -> None:
        if match not in ("normalized", "exact"):
            raise ValueError(f"match must be 'normalized' or 'exact', got {match!r}")
        self.session_id = session_id or secrets.token_hex(8)
        self.match = match
        self._secret = (secret or secrets.token_hex(32)).encode("utf-8")
        self._by_key: dict[tuple[str, str], Pseudonym] = {}
        self._by_bare: dict[str, Pseudonym] = {}
        self._counters: dict[str, int] = {}
        self._variants: dict[str, list[str]] = {}
        self._by_index: dict[tuple[str, int], Pseudonym] = {}

    def pseudonymize(self, original: str, entity_type: str) -> Pseudonym:
        """Return the placeholder for ``original``, minting one if needed."""
        key = (entity_type, self._match_key(original))
        existing = self._by_key.get(key)
        if existing is not None:
            self._remember_variant(existing, original)
            return existing

        self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
        pseudonym = Pseudonym(
            placeholder=PLACEHOLDER_TEMPLATE.format(
                entity_type=entity_type,
                index=self._counters[entity_type],
            ),
            entity_type=entity_type,
            original=original,
            # Always fingerprint the normalized value, even in "exact" mode:
            # two spellings then still fingerprint alike, so the link between
            # them survives locally even when their placeholders differ.
            fingerprint=self._fingerprint((entity_type, _normalize(original))),
        )
        self._by_key[key] = pseudonym
        self._by_bare[pseudonym.bare] = pseudonym
        self._by_index[(entity_type.upper(), self._counters[entity_type])] = pseudonym
        self._variants[pseudonym.bare] = [original]
        return pseudonym

    def variants(self, pseudonym: Pseudonym | str) -> list[str]:
        """Every distinct spelling that mapped to this placeholder.

        More than one entry means restoration canonicalized something: those
        mentions go back into the text spelled like the first one.
        """
        bare = pseudonym.bare if isinstance(pseudonym, Pseudonym) else pseudonym.strip("<>")
        return list(self._variants.get(bare, []))

    def canonicalized(self) -> list[Pseudonym]:
        """Pseudonyms that saw more than one spelling of their value."""
        return [p for p in self._by_key.values() if len(self._variants[p.bare]) > 1]

    def resolve(self, placeholder: str) -> Pseudonym | None:
        """Look up a placeholder, with or without its angle brackets."""
        return self._by_bare.get(placeholder.strip("<>"))

    def restore(self, text: str) -> str:
        """Replace every known placeholder in ``text`` with its real value.

        Matching is deliberately forgiving. The system prompt asks the model
        to echo tokens verbatim, and models routinely do not: they title-case
        them, swap the underscore for a space or a hyphen, drop the leading
        zeros, or wrap them in markdown. A token that comes back mangled and
        fails to resolve is the worst outcome this tool has -- the user reads
        "Person_001" where a name should be -- so anything placeholder-shaped
        is parsed and looked up by (type, number).

        Being liberal costs nothing: a shape that resolves to no entry is left
        exactly as it was found.
        """
        if not self._by_index:
            return text
        return _PLACEHOLDER_SHAPE.sub(self._restore_one, text)

    def _restore_one(self, match: re.Match[str]) -> str:
        key = (match.group("type").upper(), int(match.group("index")))
        pseudonym = self._by_index.get(key)
        return pseudonym.original if pseudonym is not None else match.group(0)

    def entries(self) -> list[Pseudonym]:
        """Every pseudonym held, in the order it was minted."""
        return list(self._by_key.values())

    def _match_key(self, original: str) -> str:
        return _normalize(original) if self.match == "normalized" else original

    def _remember_variant(self, pseudonym: Pseudonym, original: str) -> None:
        seen = self._variants[pseudonym.bare]
        if original not in seen:
            seen.append(original)

    def _fingerprint(self, key: tuple[str, str]) -> str:
        digest = hmac.new(self._secret, f"{key[0]}:{key[1]}".encode("utf-8"), sha256)
        return digest.hexdigest()[:12]

    def __len__(self) -> int:
        return len(self._by_key)


def _normalize(value: str) -> str:
    """Collapse whitespace and case so "John  SMITH" == "john smith"."""
    return " ".join(value.split()).casefold()
