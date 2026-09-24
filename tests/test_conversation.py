"""Multi-turn conversation, and the masked transcript that carries it.

The model remembers nothing: every turn resends the whole thread. What is
resent must be the masked transcript, because a restored answer placed in
the history would leak on this turn and on every turn after it.
"""

from helpers import FakeLLM, ValueDetector

from anonagent.agent.pipeline import PrivacyPipeline
from anonagent.privacy.masker import Masker

NAME = "James Bond"
FIRST = f"Does {NAME} have clearance?"
SECOND = "And when did he get it?"


def build(reply="Yes, <PERSON_001> has had it since March.", **kwargs):
    masker = Masker(detector=ValueDetector([(NAME, "PERSON")]))
    llm = FakeLLM(reply)
    return llm, PrivacyPipeline(llm, masker, **kwargs)


def test_the_first_turn_sends_no_history():
    llm, pipeline = build()
    pipeline.ask(FIRST)
    assert len(llm.calls[-1]) == 2  # system + question


def test_the_second_turn_replays_the_first():
    llm, pipeline = build()
    pipeline.ask(FIRST)
    pipeline.ask(SECOND)

    payload = llm.last_payload
    assert "<PERSON_001>" in payload
    assert "has had it since March" in payload, "the model's own reply must come back"
    assert len(llm.calls[-1]) == 4  # system + Q1 + A1 + Q2


def test_no_original_value_crosses_the_network_on_any_turn():
    """The guarantee, held across a conversation rather than one question."""
    llm, pipeline = build()
    for question in (FIRST, SECOND, f"Was {NAME} notified?"):
        pipeline.ask(question)

    for call in llm.calls:
        payload = "\n".join(str(message.content) for message in call)
        assert NAME not in payload


def test_the_history_holds_the_masked_answer_not_the_restored_one():
    """One wrong variable here leaks the real data on every later turn."""
    llm, pipeline = build()
    answer = pipeline.ask(FIRST)

    assert NAME in answer.answer, "the user does see the real name"
    transcript = "\n".join(str(message.content) for message in pipeline.history)
    assert NAME not in transcript
    assert "<PERSON_001>" in transcript


def test_the_thread_is_trimmed_to_the_turn_limit():
    llm, pipeline = build(max_turns=2)
    for index in range(5):
        pipeline.ask(f"Question {index} about {NAME}?")

    assert len(pipeline.history) == 4  # two turns, question and answer each
    assert "Question 4" in str(pipeline.history[-2].content)
    assert "Question 0" not in str(pipeline.history[0].content)


def test_resetting_forgets_the_thread_but_keeps_the_placeholders():
    llm, pipeline = build()
    pipeline.ask(FIRST)
    pipeline.reset_conversation()

    assert pipeline.history == []
    answer = pipeline.ask(f"Is {NAME} still active?")
    assert answer.masked_question == "Is <PERSON_001> still active?", (
        "the same person must keep the same placeholder after a reset"
    )


def test_events_belong_to_their_own_turn():
    """The UI renders one turn at a time, not the whole session so far."""
    llm, pipeline = build()
    first = pipeline.ask(FIRST)
    second = pipeline.ask(SECOND)

    assert [e.step for e in first.events] == [e.step for e in second.events]
    assert len(pipeline.log) == len(first.events) + len(second.events)


def test_a_new_person_in_a_later_turn_gets_the_next_placeholder():
    masker = Masker(detector=ValueDetector([(NAME, "PERSON"), ("Maria Whitfield", "PERSON")]))
    llm = FakeLLM("ok")
    pipeline = PrivacyPipeline(llm, masker)

    pipeline.ask(FIRST)
    answer = pipeline.ask("Did Maria Whitfield approve it?")

    assert answer.masked_question == "Did <PERSON_002> approve it?"


def test_the_request_event_reports_how_much_history_went_along():
    llm, pipeline = build()
    pipeline.ask(FIRST)
    second = pipeline.ask(SECOND)

    request = next(e for e in second.events if e.step == "llm_request")
    assert request.payload["history_messages"] == 2


def chat_session(lines, reply="Yes, <PERSON_001> has had it since March."):
    """Drive the CLI chat loop with scripted input and capture what it prints."""
    import io

    from anonagent.cli import _chat

    llm, pipeline = build(reply)
    code = _chat(pipeline, io.StringIO("".join(f"{line}\n" for line in lines)))
    return llm, pipeline, code


def test_chat_runs_a_turn_per_line(capsys):
    llm, pipeline, code = chat_session([FIRST, SECOND])
    output = capsys.readouterr().out

    assert code == 0
    assert len(llm.calls) == 2
    assert output.count("-> cloud sees:") == 2


def test_chat_shows_the_user_real_names_and_the_cloud_placeholders(capsys):
    chat_session([FIRST])
    output = capsys.readouterr().out

    sent = next(line for line in output.splitlines() if "cloud sees" in line)
    shown = next(line for line in output.splitlines() if line.strip().startswith("<-"))
    assert NAME not in sent
    assert NAME in shown


def test_chat_ends_on_a_blank_line(capsys):
    llm, _, code = chat_session(["", FIRST])
    assert (code, llm.calls) == (0, [])


def test_chat_ends_on_exit(capsys):
    llm, _, code = chat_session([FIRST, "exit", SECOND])
    assert code == 0
    assert len(llm.calls) == 1


def test_chat_ends_at_end_of_input(capsys):
    llm, _, code = chat_session([FIRST])
    assert code == 0
    assert len(llm.calls) == 1


def test_a_failed_turn_does_not_end_the_chat(capsys):
    import io

    from anonagent.agent.pipeline import ModelUnavailable
    from anonagent.cli import _chat

    llm, pipeline = build()
    calls = {"n": 0}
    original = pipeline.ask

    def flaky(question):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ModelUnavailable("no connection to the provider")
        return original(question)

    pipeline.ask = flaky
    code = _chat(pipeline, io.StringIO(f"{FIRST}\n{SECOND}\n"))

    assert code == 0
    assert calls["n"] == 2, "the second question must still be asked"
    assert "Could not reach the model" in capsys.readouterr().err
