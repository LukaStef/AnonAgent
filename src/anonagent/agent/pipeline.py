"""The round trip: mask locally, reason in the cloud, restore locally.

Every step is recorded into an EventLog so the console and the UI can show
what actually happened rather than a summary of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from anonagent.events import AgentEvent, EventLog
from anonagent.privacy.masker import Masker, MaskingResult

SYSTEM_PROMPT = """You are assisting with text in which every private value \
has been replaced by an opaque placeholder such as <PERSON_001>, \
<EMAIL_ADDRESS_002> or <IBAN_CODE_001>.

Work by these rules:
- Each placeholder is a stable identifier for one real entity. <PERSON_001> \
is the same person everywhere it appears.
- Reason about the placeholders as you would about named entities. Do not \
speculate about which real person, address or number one stands for.
- When you refer to an entity, reproduce its placeholder exactly as written, \
angle brackets included.
- Never invent a placeholder, renumber an existing one, or substitute a \
plausible real value for one.
- If a question genuinely cannot be answered without the hidden value, say \
which value is missing and why it matters."""


class ModelUnavailable(RuntimeError):
    """The cloud model could not be reached, or refused the request.

    Raised in place of whatever the provider's SDK threw, so a dropped wifi
    connection reads as one sentence instead of a stack trace. Subclasses
    RuntimeError, which the CLI already treats as a message to print.
    """


class LeakDetected(RuntimeError):
    """A private value was about to be sent to the cloud model.

    The pipeline fails closed on this: refusing to answer is always better
    than answering at the cost of the guarantee the tool exists to provide.
    """


@dataclass(frozen=True)
class Answer:
    """One question, answered through the privacy layer."""

    question: str
    masked_question: str
    masked_answer: str
    answer: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    events: list[AgentEvent] = field(default_factory=list)


#: Exchanges kept when replaying the conversation. Every turn resends the
#: whole history, so this caps both the token bill and the context window.
DEFAULT_MAX_TURNS = 8


class PrivacyPipeline:
    """Masks a question, asks the cloud model, restores the answer.

    The model remembers nothing between calls: a conversation is the client
    resending everything each turn. What gets resent here is the *masked*
    transcript, so the provider follows a coherent thread about <PERSON_001>
    while never learning a name.
    """

    def __init__(
        self,
        llm: Any,
        masker: Masker | None = None,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        log: EventLog | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self.llm = llm
        self.masker = Masker() if masker is None else masker
        self.system_prompt = system_prompt
        self.log = EventLog() if log is None else log
        self.max_turns = max_turns
        #: Masked messages only. Putting a restored answer in here would send
        #: the real values back out on every subsequent turn.
        self.history: list[BaseMessage] = []

    def reset_conversation(self) -> None:
        """Forget the thread, keep the vault.

        Placeholders stay stable across a reset, so the same person is still
        <PERSON_001> in the next conversation.
        """
        self.history = []

    def ask(self, question: str) -> Answer:
        """Mask the question, ask the model, restore the answer."""
        turn_start = len(self.log)
        self.log.record(
            "input_received",
            "Question received locally",
            {"text": question},
            sensitive=True,
        )
        return self._ask_masked(self.masker.mask(question), turn_start)

    def ask_masked(self, result: MaskingResult) -> Answer:
        """Continue from a masking a caller has already done.

        A screen that shows the pipeline step by step needs the detection and
        the pseudonyms before the network call, not after it. Handing the
        result back avoids masking the same text twice, which would also
        double-count what the sweep caught the second time round.
        """
        turn_start = len(self.log)
        self.log.record(
            "input_received",
            "Question received locally",
            {"text": result.original_text},
            sensitive=True,
        )
        return self._ask_masked(result, turn_start)

    def _ask_masked(self, result: MaskingResult, turn_start: int) -> Answer:
        question = result.original_text
        self.log.record(
            "pii_detected",
            f"Found {len(result.entities)} sensitive spans",
            {"counts": result.entity_counts},
        )
        if result.sealed:
            self.log.record(
                "sealed",
                f"Sweep caught {len(result.sealed)} value(s) the detector missed",
                {"count": len(result.sealed)},
            )

        self._guard(result.masked_text)
        self.log.record(
            "masked",
            "Question rewritten with placeholders",
            {"text": result.masked_text},
        )

        masked_answer = self._invoke(result.masked_text)
        self._remember(result.masked_text, masked_answer)
        self.log.record(
            "llm_response",
            "Cloud model answered",
            {"text": masked_answer},
        )

        answer = self.masker.unmask(masked_answer)
        self.log.record(
            "unmasked",
            "Placeholders restored locally",
            {"text": answer},
            sensitive=True,
        )

        return Answer(
            question=question,
            masked_question=result.masked_text,
            masked_answer=masked_answer,
            answer=answer,
            entity_counts=result.entity_counts,
            events=self.log.events[turn_start:],
        )

    def _remember(self, masked_question: str, masked_answer: str) -> None:
        """Append this turn to the transcript, masked, and trim the oldest."""
        self.history.append(HumanMessage(content=masked_question))
        self.history.append(AIMessage(content=masked_answer))
        excess = len(self.history) - self.max_turns * 2
        if excess > 0:
            del self.history[:excess]

    def _guard(self, masked_text: str) -> None:
        """Refuse to send anything holding a value the vault knows is private."""
        leaked = self.masker.leaks(masked_text)
        if leaked:
            self.log.record(
                "leak_blocked",
                f"Refused to send: {len(leaked)} private value(s) still present",
                {"count": len(leaked)},
            )
            raise LeakDetected(
                f"{len(leaked)} private value(s) survived masking; refusing to send"
            )
        self.log.record("leak_check", "Verified: no private value in the outbound text")

    def _invoke(self, masked_text: str) -> str:
        self.log.record(
            "llm_request",
            "Sending masked text to the cloud model",
            {
                "text": masked_text,
                "model": _model_name(self.llm),
                "history_messages": len(self.history),
            },
        )
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.history)
        messages.append(HumanMessage(content=masked_text))
        try:
            reply = self.llm.invoke(messages)
        except Exception as error:  # noqa: BLE001 -- every provider raises its own
            # The provider's own message is third-party text: it echoes
            # request content back and we do not control what ends up in it.
            reason = self.masker.redact(_describe_failure(error))
            self.log.record("llm_failed", f"Cloud model unreachable: {reason}")
            raise ModelUnavailable(reason) from error

        return getattr(reply, "content", reply) if not isinstance(reply, str) else reply


#: Matched against the provider exception's class name. Every SDK we support
#: names its failures this way, so we classify without importing any of them
#: and without a branch per provider.
_FAILURES: tuple[tuple[str, str], ...] = (
    ("Authentication", "the API key was rejected. Check the key in .env"),
    ("PermissionDenied", "the API key is not allowed to use this model"),
    ("RateLimit", "the provider is rate limiting us. Wait a moment, or use a smaller model"),
    ("NotFound", "the provider does not know that model name"),
    ("Connection", "no connection to the provider. Check the network"),
    ("Timeout", "the provider did not answer in time"),
    ("BadRequest", "the provider rejected the request as malformed"),
    ("InternalServer", "the provider is having trouble on its end"),
)


def _describe_failure(error: Exception) -> str:
    """Turn a provider exception into a sentence a person can act on."""
    name = type(error).__name__
    for marker, explanation in _FAILURES:
        if marker.lower() in name.lower():
            return explanation
    detail = str(error).strip().splitlines()[0] if str(error).strip() else name
    return f"{name}: {detail}"


def _model_name(llm: Any) -> str:
    for attribute in ("model_name", "model"):
        name = getattr(llm, attribute, None)
        if isinstance(name, str):
            return name
    return type(llm).__name__
