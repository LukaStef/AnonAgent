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


def test_the_page_admits_it_misses_things(app):
    """A tool that implies it catches everything is worse than one that says so."""
    page = rendered(app)
    assert "miss things" in page
    assert "middle column" in page


def drive(confirm_first: bool):
    """Run the page with no key in the environment, optionally holding first."""
    with pytest.MonkeyPatch.context() as environment:
        for variable in ("OPENAI_API_KEY", "GROQ_API_KEY", "OLLAMA_BASE_URL"):
            environment.setenv(variable, "")
        at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=180)
        at.run()
        if confirm_first:
            at.checkbox(key="confirm_first").set_value(True).run()
        at.button[0].click().run()
        return at


def test_checking_the_box_stops_before_the_network():
    """The whole point: the payload is readable while it is still local."""
    at = drive(confirm_first=True)
    page = rendered(at)

    assert "waiting for you to confirm" in page
    assert "Nothing has been sent" in page
    assert "Send it" in [button.label for button in at.button]


def test_the_payload_is_on_screen_while_it_waits():
    at = drive(confirm_first=True)
    cards = rendered(at, only_cards=True)

    # The card markup is escaped, so the brackets arrive as entities.
    assert "PERSON_001" in cards
    for value in PRIVATE:
        assert value not in cards


def test_leaving_the_box_unchecked_does_not_stop():
    at = drive(confirm_first=False)
    page = rendered(at)

    assert "Nothing has been sent" not in page
    assert "Send it" not in [button.label for button in at.button]


def test_a_finished_exchange_survives_the_next_click():
    """Regression: the answer used to vanish on any later interaction.

    Streamlit reruns the whole script on every click, so anything rendered
    during a previous run is gone unless it is redrawn from state.
    """
    at = drive(confirm_first=True)
    send = [b for b in at.button if b.label == "Send it"][0]
    send.click().run()
    after_send = rendered(at)

    # Any unrelated interaction: toggling the checkbox reruns the script.
    at.checkbox(key="confirm_first").set_value(False).run()
    after_rerun = rendered(at)

    assert "PERSON_001" in after_send
    assert "PERSON_001" in after_rerun, "the panels emptied themselves"


def test_discarding_clears_the_screen():
    at = drive(confirm_first=True)
    [b for b in at.button if b.label == "Discard"][0].click().run()
    page = rendered(at)

    assert "Nothing has been sent" not in page
    assert "waiting for input" in page
