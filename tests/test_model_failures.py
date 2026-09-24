"""What the user sees when the cloud model cannot be reached.

A dropped connection mid-demo is the likeliest failure this tool has, and it
has nothing to do with privacy. It must read as one sentence, not a stack
trace, and it must never take the private data down with it.
"""

import pytest
from helpers import StubDetector, span

from anonagent.agent.pipeline import ModelUnavailable, PrivacyPipeline
from anonagent.privacy.masker import Masker

QUESTION = "Does Nikola Stefanovic have clearance?"


class FailingLLM:
    """Raises whatever a provider SDK would raise."""

    model_name = "fake-model"

    def __init__(self, error):
        self.error = error

    def invoke(self, messages):
        raise self.error


def pipeline_raising(error):
    detector = StubDetector([span(QUESTION, "Nikola Stefanovic", "PERSON")])
    return PrivacyPipeline(FailingLLM(error), Masker(detector=detector))


def named_error(name, message="boom"):
    """A stand-in for a provider exception, identified by its class name."""
    return type(name, (Exception,), {})(message)


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("APIConnectionError", "no connection"),
        ("APITimeoutError", "did not answer in time"),
        ("RateLimitError", "rate limiting"),
        ("AuthenticationError", "key was rejected"),
        ("PermissionDeniedError", "not allowed"),
        ("NotFoundError", "does not know that model"),
        ("BadRequestError", "malformed"),
        ("InternalServerError", "trouble on its end"),
    ],
)
def test_a_provider_failure_becomes_a_readable_sentence(class_name, expected):
    pipeline = pipeline_raising(named_error(class_name))

    with pytest.raises(ModelUnavailable, match=expected):
        pipeline.ask(QUESTION)


def test_an_unrecognized_failure_still_says_something_useful():
    pipeline = pipeline_raising(named_error("WeirdProviderError", "socket exploded"))

    with pytest.raises(ModelUnavailable, match="socket exploded"):
        pipeline.ask(QUESTION)


def test_the_original_exception_is_kept_for_debugging():
    original = named_error("APIConnectionError")
    pipeline = pipeline_raising(original)

    with pytest.raises(ModelUnavailable) as caught:
        pipeline.ask(QUESTION)
    assert caught.value.__cause__ is original


def test_the_failure_is_recorded_in_the_log():
    pipeline = pipeline_raising(named_error("APIConnectionError"))

    with pytest.raises(ModelUnavailable):
        pipeline.ask(QUESTION)

    steps = [e.step for e in pipeline.log.events]
    assert steps[-1] == "llm_failed"
    assert "unmasked" not in steps


def test_a_failure_message_never_carries_the_private_value():
    """An error string is the easiest place for a secret to escape."""
    pipeline = pipeline_raising(named_error("WeirdError", "failed on Nikola Stefanovic"))

    with pytest.raises(ModelUnavailable) as caught:
        pipeline.ask(QUESTION)

    assert "Nikola Stefanovic" not in str(caught.value)


def test_a_multiline_provider_message_is_trimmed_to_one_line():
    pipeline = pipeline_raising(named_error("WeirdError", "first line\nstack\nmore stack"))

    with pytest.raises(ModelUnavailable) as caught:
        pipeline.ask(QUESTION)

    assert "\n" not in str(caught.value)
