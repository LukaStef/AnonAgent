"""Shared test doubles.

The detector and the model are stubbed so the suite stays fast, offline and
deterministic: these tests check our logic, not spaCy's recall or OpenAI's
wording.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from anonagent.privacy.detector import DetectedEntity
from anonagent.privacy.masker import Masker


class StubDetector:
    """Reports exactly the spans it was given, if the text still matches."""

    def __init__(self, entities):
        self._entities = entities

    def detect(self, text):
        return [e for e in self._entities if text[e.start : e.end] == e.text]


class FakeLLM:
    """A chat model that returns a canned reply and remembers what it was sent."""

    model_name = "fake-model"

    def __init__(self, reply: str = "Acknowledged."):
        self.reply = reply
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply)

    @property
    def last_payload(self) -> str:
        """Everything that would have crossed the network on the last call."""
        return "\n".join(str(message.content) for message in self.calls[-1])


class LeakyMasker(Masker):
    """A masker with the safety sweep disabled, to test the guard behind it."""

    def _sweep(self, masked: str):
        return masked, []


class ValueDetector:
    """Finds given values wherever they appear in whatever text it is handed.

    Unlike StubDetector it holds no offsets, so it works on a chunk as well
    as on a whole document -- which is what document tests need.
    """

    def __init__(self, values, score: float = 0.9):
        self.values = list(values)
        self.score = score

    def detect(self, text):
        found = []
        for value, entity_type in self.values:
            start = text.find(value)
            while start != -1:
                found.append(
                    DetectedEntity(entity_type, start, start + len(value), self.score, value)
                )
                start = text.find(value, start + len(value))
        return sorted(found, key=lambda e: e.start)


def span(text: str, value: str, entity_type: str, score: float = 0.9) -> DetectedEntity:
    start = text.index(value)
    return DetectedEntity(entity_type, start, start + len(value), score, value)
