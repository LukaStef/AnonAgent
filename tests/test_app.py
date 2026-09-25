"""The screen itself.

The middle column is what gets projected. Everything in it must be safe to
show a room full of strangers, which is a promise the layout makes and only
a test can keep.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

#: AppTest resolves a relative path against the caller's directory, which is
#: tests/, so point at the app from the project root explicitly.
APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def app():
    """Drive the page with no API key in the environment.

    Without this the suite behaves differently on a machine that happens to
    have a key: the page would reach a cloud model, so the tests would be
    slow, non-deterministic, and quietly spending someone's credit.

    The variables are emptied rather than deleted, because the page calls
    load_dotenv() as it starts and that would put a deleted one straight
    back. An empty value stays empty: load_dotenv leaves alone what is
    already set.
    """
    with pytest.MonkeyPatch.context() as environment:
        for variable in ("OPENAI_API_KEY", "GROQ_API_KEY", "OLLAMA_BASE_URL"):
            environment.setenv(variable, "")
        at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        at.button[0].click().run()
        return at


def rendered(app, only_cards: bool = False) -> str:
    blocks = [m.value for m in app.markdown]
    if only_cards:
        blocks = [b for b in blocks if "card" in b]
    return "\n".join(blocks)


PRIVATE = ["James Bond", "0412987654321", "DE89370400440532013000"]


def test_the_page_renders_and_submits_without_error(app):
    # AppTest.exception is a list, empty when the script ran cleanly.
    raised = [e.value for e in app.exception]
    assert raised == []


@pytest.mark.parametrize("value", PRIVATE)
def test_the_projected_column_never_shows_a_private_value(app, value):
    assert value not in rendered(app, only_cards=True)


def test_the_projected_column_does_show_the_placeholders(app):
    cards = rendered(app, only_cards=True)
    assert "PERSON_001" in cards
    assert "IBAN_CODE_001" in cards


def test_the_local_column_still_shows_the_input(app):
    assert "James Bond" in app.text_area[0].value


def test_without_a_key_the_local_half_still_runs(app):
    """The demo has to work on a laptop with no API key at all."""
    cards = rendered(app, only_cards=True)
    assert "OPENAI_API_KEY" in cards
    assert "not set" in cards
