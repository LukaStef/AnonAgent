"""Splitting a document into pieces small enough to analyze quickly.

Detection time grows faster than the text does -- 10 KB takes a quarter of a
second, 500 KB takes nearly two minutes -- so a long document is analyzed in
pieces and the cost becomes linear again.

Where the cuts fall matters more than how big the pieces are. A value sliced
down the middle is recognized in neither half and leaks in both, so cuts are
made at paragraph breaks first, sentence ends second, and mid-sentence only
when a single sentence is longer than a whole chunk.
"""

from __future__ import annotations

import re

#: Chunk size in characters. Small enough to stay in the fast regime,
#: large enough that most documents are a handful of pieces.
DEFAULT_CHUNK_CHARS = 8_000

#: A blank line, kept as its own piece so chunks retile the original exactly.
_PARAGRAPH_BREAK = re.compile(r"(\n[ \t]*\n)")

_SENTENCE_END = re.compile(r"(?<=[.!?])(\s+)")


def split_into_chunks(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into pieces that join back into exactly ``text``.

    The pieces are no longer than ``max_chars`` unless a single unbreakable
    run of text is itself longer, in which case one piece exceeds the limit
    rather than a value being cut in half.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return [text] if text else []

    return _pack(_atoms(text, max_chars), max_chars)


def _atoms(text: str, max_chars: int) -> list[str]:
    """The smallest pieces we are willing to cut between."""
    atoms: list[str] = []
    for paragraph in _split_keeping_separators(text, _PARAGRAPH_BREAK):
        if len(paragraph) <= max_chars:
            atoms.append(paragraph)
            continue
        for sentence in _split_keeping_separators(paragraph, _SENTENCE_END):
            if len(sentence) <= max_chars:
                atoms.append(sentence)
            else:
                # One sentence longer than a whole chunk. Nothing safe is left
                # to cut on, so cut on length and accept the risk here alone.
                atoms.extend(
                    sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)
                )
    return atoms


def _split_keeping_separators(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on ``pattern``, keeping each separator attached to its left piece.

    Keeping the separators is what lets the chunks be joined back into the
    original byte for byte.
    """
    parts = pattern.split(text)
    pieces: list[str] = []
    # re.split with one capture group alternates content, separator, content...
    for index in range(0, len(parts), 2):
        content = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if content or separator:
            pieces.append(content + separator)
    return pieces


def _pack(atoms: list[str], max_chars: int) -> list[str]:
    """Greedily fill chunks up to the limit without ever splitting an atom."""
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        if current and len(current) + len(atom) > max_chars:
            chunks.append(current)
            current = atom
        else:
            current += atom
    if current:
        chunks.append(current)
    return chunks
