"""CLI input handling.

Mostly a regression guard: an earlier version read stdin whenever it was not
a terminal, which hangs forever on an open pipe with nothing in it.
"""

import argparse
import io

from anonagent.cli import _read_input
from anonagent.demo_data import DEFAULT_SAMPLE, SAMPLES


def namespace(**overrides):
    defaults = {"text": None, "sample": DEFAULT_SAMPLE, "file": None}
    return argparse.Namespace(**{**defaults, **overrides})


def test_an_argument_is_used_as_the_text():
    assert _read_input(namespace(text="Call Maria")) == "Call Maria"


def test_no_argument_falls_back_to_the_sample(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("this must not be read"))
    assert _read_input(namespace()) == SAMPLES[DEFAULT_SAMPLE]


def test_a_dash_reads_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("  piped text  \n"))
    assert _read_input(namespace(text="-")) == "piped text"


def test_a_file_argument_wins_over_everything(tmp_path):
    document = tmp_path / "record.txt"
    document.write_text("Patient Maria Whitfield.", encoding="utf-8")
    assert _read_input(namespace(file=str(document), text="ignored")) == (
        "Patient Maria Whitfield."
    )


def test_the_default_threshold_masks_more_than_it_misses():
    """Guards against someone "tidying" the default upward."""
    import inspect

    from anonagent.privacy.detector import PiiDetector

    default = inspect.signature(PiiDetector.__init__).parameters["score_threshold"].default
    assert default <= 0.5
