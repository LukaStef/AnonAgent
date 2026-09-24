"""The end-to-end round trip, with the cloud model stubbed out."""

import pytest
from helpers import FakeLLM, LeakyMasker, StubDetector, span

from anonagent.agent.pipeline import LeakDetected, PrivacyPipeline
from anonagent.privacy.masker import Masker
from anonagent.privacy.vault import PseudonymVault

QUESTION = "Does James Bond at james@example.com have clearance?"


def build(reply, question=QUESTION, masker_class=Masker):
    detector = StubDetector(
        [
            span(question, "James Bond", "PERSON"),
            span(question, "james@example.com", "EMAIL_ADDRESS"),
        ]
    )
    llm = FakeLLM(reply)
    return llm, PrivacyPipeline(llm, masker_class(detector=detector))


def test_the_answer_comes_back_with_real_values_restored():
    llm, pipeline = build("Yes, <PERSON_001> has clearance. Confirm at <EMAIL_ADDRESS_001>.")
    answer = pipeline.ask(QUESTION)

    assert answer.answer == (
        "Yes, James Bond has clearance. Confirm at james@example.com."
    )


def test_nothing_private_crosses_the_network():
    """The guarantee the whole project rests on."""
    llm, pipeline = build("Yes, <PERSON_001> has clearance.")
    pipeline.ask(QUESTION)

    payload = llm.last_payload
    assert "James Bond" not in payload
    assert "james@example.com" not in payload
    assert "<PERSON_001>" in payload


def test_the_system_prompt_travels_with_the_question():
    llm, pipeline = build("ok")
    pipeline.ask(QUESTION)

    assert "placeholder" in llm.last_payload.lower()
    assert len(llm.calls[-1]) == 2


def test_no_public_event_carries_a_private_value():
    """The middle panel of the UI renders these. It must be safe to show."""
    llm, pipeline = build("Yes, <PERSON_001> has clearance.")
    answer = pipeline.ask(QUESTION)

    public = [e for e in answer.events if not e.sensitive]
    assert public, "expected some public events"
    for event in public:
        rendered = f"{event.message} {event.payload}"
        assert "James Bond" not in rendered
        assert "james@example.com" not in rendered


def test_the_sensitive_events_are_the_first_and_the_last():
    llm, pipeline = build("Yes, <PERSON_001> has clearance.")
    answer = pipeline.ask(QUESTION)

    sensitive = [e.step for e in answer.events if e.sensitive]
    assert sensitive == ["input_received", "unmasked"]


def test_the_steps_are_recorded_in_order():
    llm, pipeline = build("ok")
    answer = pipeline.ask(QUESTION)

    assert [e.step for e in answer.events] == [
        "input_received",
        "pii_detected",
        "leak_check",
        "masked",
        "llm_request",
        "llm_response",
        "unmasked",
    ]


def test_a_question_with_nothing_private_still_works():
    question = "Summarize the quarterly revenue trend."
    llm = FakeLLM("Revenue grew.")
    pipeline = PrivacyPipeline(llm, Masker(detector=StubDetector([])))

    answer = pipeline.ask(question)
    assert answer.masked_question == question
    assert answer.answer == "Revenue grew."
    assert answer.entity_counts == {}


def test_the_model_inventing_an_unknown_placeholder_is_left_alone():
    """Better a visible <PERSON_009> than a confidently wrong real name."""
    llm, pipeline = build("<PERSON_001> reports to <PERSON_009>.")
    answer = pipeline.ask(QUESTION)

    assert answer.answer == "James Bond reports to <PERSON_009>."


def test_entity_counts_are_reported():
    llm, pipeline = build("ok")
    answer = pipeline.ask(QUESTION)
    assert answer.entity_counts == {"PERSON": 1, "EMAIL_ADDRESS": 1}


def test_the_guard_refuses_to_send_a_value_that_survived_masking():
    question = "James Bond again: James Bond."
    detector = StubDetector([span(question, "James Bond", "PERSON")])
    llm = FakeLLM("ok")
    pipeline = PrivacyPipeline(llm, LeakyMasker(detector=detector))

    with pytest.raises(LeakDetected):
        pipeline.ask(question)

    assert llm.calls == [], "nothing may be sent once a leak is detected"


def test_a_blocked_leak_is_recorded():
    question = "James Bond again: James Bond."
    detector = StubDetector([span(question, "James Bond", "PERSON")])
    pipeline = PrivacyPipeline(FakeLLM("ok"), LeakyMasker(detector=detector))

    with pytest.raises(LeakDetected):
        pipeline.ask(question)

    steps = [e.step for e in pipeline.log.events]
    assert "leak_blocked" in steps
    assert "llm_request" not in steps


def test_one_vault_spans_several_questions():
    """A follow-up question must reuse the placeholders of the first."""
    vault = PseudonymVault()
    first = "James Bond filed it."
    second = "When did James Bond file it?"

    detector = StubDetector(
        [span(first, "James Bond", "PERSON"), span(second, "James Bond", "PERSON")]
    )
    llm = FakeLLM("<PERSON_001> filed it on Tuesday.")
    pipeline = PrivacyPipeline(llm, Masker(detector=detector, vault=vault))

    pipeline.ask(first)
    answer = pipeline.ask(second)

    assert answer.masked_question == "When did <PERSON_001> file it?"
    assert len(vault) == 1
