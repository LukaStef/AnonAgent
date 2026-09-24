"""Asking the user about detections the detector is unsure of.

A low score means the detector found something that may or may not be
private. Guessing either way is wrong: masking everything ruins the answer,
masking nothing leaks. So below a confidence threshold, the person who owns
the data decides.

Every default here fails safe. An empty answer masks, an unreadable stream
masks, a non-interactive session masks. Silence never means "send it".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:  # importing it for real makes a cycle through privacy/__init__
    from anonagent.privacy.detector import DetectedEntity

#: Detections at or above this score are masked without asking.
DEFAULT_CONFIRM_BELOW = 0.5

Decision = tuple[list["DetectedEntity"], list["DetectedEntity"]]


def mask_everything(entities: list[DetectedEntity]) -> Decision:
    """The default reviewer: every detection is masked, nothing is asked."""
    return entities, []


@dataclass
class InteractiveReviewer:
    """Masks confident detections, asks about the rest.

    A decision is remembered per value, so a name appearing eight times is
    asked about once.
    """

    confirm_below: float = DEFAULT_CONFIRM_BELOW
    input_stream: TextIO | None = None
    output_stream: TextIO | None = None
    interactive: bool | None = None
    _decisions: dict[tuple[str, str], bool] = field(default_factory=dict, init=False)

    def __call__(self, entities: list[DetectedEntity]) -> Decision:
        approved: list[DetectedEntity] = []
        declined: list[DetectedEntity] = []
        for entity in entities:
            if entity.score >= self.confirm_below or self._approve(entity):
                approved.append(entity)
            else:
                declined.append(entity)
        return approved, declined

    def _approve(self, entity: DetectedEntity) -> bool:
        key = (entity.entity_type, entity.text.casefold())
        if key not in self._decisions:
            self._decisions[key] = self._ask(entity)
        return self._decisions[key]

    def _ask(self, entity: DetectedEntity) -> bool:
        out = self.output_stream or sys.stdout
        if not self._is_interactive():
            print(
                f"  uncertain {entity.entity_type} {entity.text!r} "
                f"(score {entity.score:.2f}) -- masking, nobody here to ask",
                file=out,
            )
            return True

        print(
            f"\n  Uncertain: {entity.entity_type} at score {entity.score:.2f}\n"
            f"    {entity.text!r}\n"
            f"  Mask it? [Y/n] ",
            end="",
            file=out,
            flush=True,
        )
        answer = (self.input_stream or sys.stdin).readline()
        if not answer:  # stream closed mid-question
            print("(no answer -- masking)", file=out)
            return True

        declined = answer.strip().casefold() in ("n", "no")
        print("kept in the clear" if declined else "masked", file=out)
        return not declined

    def _is_interactive(self) -> bool:
        if self.interactive is not None:
            return self.interactive
        stream = self.input_stream or sys.stdin
        return bool(getattr(stream, "isatty", lambda: False)())
