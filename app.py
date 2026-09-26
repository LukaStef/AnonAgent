"""AnonAgent, on screen.

Three columns, mirroring the three places the data lives: what you typed and
never leaves this machine, what the agent did with it, and the answer with
your values put back. The middle column is the point of the whole thing --
everything in it is safe to show an audience, and a test enforces that.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from anonagent.agent.llm import DEFAULT_MODELS, build_llm
from anonagent.agent.pipeline import LeakDetected, ModelUnavailable, PrivacyPipeline
from anonagent.privacy import Masker, PiiDetector, PseudonymVault

load_dotenv()

st.set_page_config(
    page_title="AnonAgent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SAMPLE = """Please analyse the following transaction for potential fraud risk:

Full name: James Bond
ID number: 0412987654321
Transaction: EUR 4,850 wire transfer to IBAN DE89370400440532013000
on 2024-11-14 at 14:32 CET.

Flag any anomalies and recommend what to do next."""

STEPS = [
    ("1", "PII detection", "waiting for input"),
    ("2", "Local pseudonymization", "waiting for the scan"),
    ("3", "Cloud payload (anonymized)", "waiting for pseudonyms"),
    ("4", "Cloud model response", "waiting for the cloud"),
]

CSS = """
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container {padding: 1.1rem 1.6rem 2rem; max-width: 100%;}

  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid #1b2430; padding-bottom: .7rem; margin-bottom: 1.1rem;
  }
  .brand {display: flex; align-items: center; gap: .55rem;
          font-size: 1.05rem; font-weight: 700; color: #e6edf5; letter-spacing: .01em;}
  .brand .mark {color: #2dd4bf; font-size: 1.15rem;}
  .status {display: flex; align-items: center; gap: .45rem;
           font-size: .78rem; color: #8b9bb0;}
  .dot {width: 7px; height: 7px; border-radius: 50%; background: #2dd4bf;
        box-shadow: 0 0 7px #2dd4bf;}

  .colhead {border-left: 3px solid #2dd4bf; padding-left: .6rem; margin-bottom: .8rem;}
  .colhead .t {font-size: .95rem; font-weight: 700; color: #e6edf5;}
  .colhead .s {font-size: .74rem; color: #6f8096;}

  .card {
    border: 1px solid #1b2430; border-radius: 8px; background: #0b1016;
    padding: .7rem .85rem; margin-bottom: .6rem;
  }
  .card .n {
    display: inline-block; width: 1.3rem; height: 1.3rem; line-height: 1.3rem;
    text-align: center; border-radius: 4px; background: #131c26;
    color: #6f8096; font-size: .72rem; margin-right: .5rem;
  }
  .card .h {font-size: .83rem; color: #c9d4e0; font-weight: 600;}
  .card.done {border-color: #1d5f56;}
  .card.done .n {background: #123029; color: #2dd4bf;}
  .card .body {
    margin-top: .5rem; font-size: .76rem; color: #7f8fa3;
    white-space: pre-wrap; word-break: break-word; line-height: 1.5;
  }
  .card .body.idle {color: #4a5a6d;}

  .chips {display: flex; flex-wrap: wrap; gap: .4rem;
          margin: .45rem 0 1.1rem;}
  .chip {
    font-size: .68rem; padding: .22rem .5rem; border-radius: 4px;
    border: 1px solid #1d5f56; background: #0f1f1c; color: #2dd4bf;
    letter-spacing: .04em;
  }
  .chip.clear {border-color: #3d3520; background: #1a1710; color: #c9a227;}
  .chip.none {border-color: #1b2430; background: #0b1016; color: #55657a;}

  .note {font-size: .72rem; color: #6f8096; margin-top: .9rem; line-height: 1.5;}
  .note .clear {color: #c9a227;}

  .result {
    border: 1px solid #1b2430; border-radius: 8px; background: #0b1016;
    padding: .85rem; min-height: 16rem; font-size: .83rem; color: #d7e2ee;
    white-space: pre-wrap; line-height: 1.6;
  }
  .result.idle {color: #4a5a6d; display: flex; align-items: center;
                justify-content: center; text-align: center;}

  .gate {
    border: 1px solid #3d3520; background: #1a1710; border-radius: 6px;
    padding: .55rem .7rem; margin: .2rem 0 .5rem;
    font-size: .74rem; color: #c9a227; line-height: 1.5;
  }

  .disclaimer {
    border-top: 1px solid #1b2430; margin-top: 1.6rem; padding-top: .8rem;
    font-size: .72rem; color: #6f8096; line-height: 1.7;
  }
  .disclaimer b {color: #c9a227; font-weight: 600;}

  .stTextArea textarea {
    background: #0b1016 !important; border: 1px solid #1b2430 !important;
    font-size: .82rem !important; line-height: 1.55 !important; color: #d7e2ee !important;
  }
  .stButton button {
    width: 100%; border: 1px solid #1d5f56; background: #0f1f1c; color: #2dd4bf;
    font-weight: 700; letter-spacing: .02em; padding: .55rem;
  }
  .stButton button:hover {border-color: #2dd4bf; background: #123029; color: #7ff0e0;}
</style>
"""


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    settings = sidebar()
    topbar(settings)

    left, middle, right = st.columns([1, 1.15, 1], gap="medium")

    with left:
        column_head("Input / raw data", "Stays on this machine")
        text = st.text_area("query", value=SAMPLE, height=300, label_visibility="collapsed")
        chips_slot = st.empty()
        confirm_first = st.checkbox(
            "Let me read the payload before it goes",
            value=False,
            key="confirm_first",
            help="Masks locally and stops. Nothing reaches the network until "
            "you have seen exactly what would be sent.",
        )
        send_column, reset_column = st.columns([3, 1])
        with send_column:
            submit = st.button("Mask and send")
        with reset_column:
            # The vault deliberately outlives a single submission, so that a
            # follow-up question keeps the same placeholders. When the text is
            # edited instead, yesterday's people linger, hence a way out.
            reset = st.button("New", help="Empty the vault and start fresh")
        mapping_slot = st.empty()
        gate_slot = st.container()

    if reset:
        st.session_state.pop("masker", None)
        st.session_state.pop("signature", None)
        st.rerun()

    with middle:
        column_head("Agent execution log", "Everything here is safe to show")
        step_slots = [st.empty() for _ in STEPS]

    with right:
        column_head("Local result", "Your values put back")
        result_slot = st.empty()

    for slot, (number, title, idle) in zip(step_slots, STEPS):
        slot.markdown(step_card(number, title, idle, done=False), unsafe_allow_html=True)
    result_slot.markdown(
        "<div class='result idle'>the answer appears here,<br>with real values restored</div>",
        unsafe_allow_html=True,
    )
    chips_slot.markdown(detected_chips(None, settings), unsafe_allow_html=True)

    masker = session_masker(settings)
    pending = st.session_state.get("pending")
    done = st.session_state.get("done")

    if submit and text.strip():
        result = masker.mask(text)
        render_local(result, settings, masker, chips_slot, step_slots, mapping_slot)
        st.session_state["done"] = done = None
        if confirm_first:
            st.session_state["pending"] = pending = result
        else:
            st.session_state["pending"] = pending = None
            answer = send_to_cloud(result, settings, masker, step_slots, result_slot)
            st.session_state["done"] = done = (result, answer)
    elif pending is not None:
        # A rerun while a payload waits: keep the panels as the user left them.
        render_local(pending, settings, masker, chips_slot, step_slots, mapping_slot)
    elif done is not None:
        # Streamlit reruns on every click, so a finished exchange has to be
        # redrawn from memory or the screen empties itself.
        redraw(done, settings, masker, chips_slot, step_slots, result_slot, mapping_slot)

    if pending is not None:
        hold(gate_slot, pending, settings, masker, step_slots, result_slot)

    disclaimer()


def hold(container, result, settings, masker, step_slots, result_slot) -> None:
    """Stop before the network and let the user read what would be sent."""
    step_slots[3].markdown(
        step_card("4", "Cloud model response", "waiting for you to confirm", done=False),
        unsafe_allow_html=True,
    )
    with container:
        st.markdown(
            "<div class='gate'>Nothing has been sent. Step 3 is exactly what "
            "would go.</div>",
            unsafe_allow_html=True,
        )
        go_column, drop_column = st.columns(2)
        if go_column.button("Send it", type="primary"):
            st.session_state["pending"] = None
            answer = send_to_cloud(result, settings, masker, step_slots, result_slot)
            st.session_state["done"] = (result, answer)
        if drop_column.button("Discard"):
            st.session_state["pending"] = None
            st.session_state["done"] = None
            st.rerun()


def disclaimer() -> None:
    """Say plainly that detection misses things.

    Whether a name is found depends on the sentence around it: the same name
    can be caught in one line and missed in the next. A tool that quietly
    implies otherwise is worse than one that says so.
    """
    st.markdown(
        "<div class='disclaimer'>"
        "<b>This will miss things.</b> Detection is statistical, so whether a "
        "value is found depends on the words around it &mdash; the same name "
        "can be caught in one sentence and missed in the next. Names the "
        "model has not seen before are the usual gap.<br>"
        "Read the cloud payload in the middle column before you trust it. "
        "Lowering the detection threshold or turning on review in the sidebar "
        "catches more, at the cost of masking things that did not need it."
        "</div>",
        unsafe_allow_html=True,
    )


def redraw(done, settings, masker, chips_slot, step_slots, result_slot, mapping_slot) -> None:
    """Put a finished exchange back on screen after an unrelated rerun."""
    result, answer = done
    render_local(result, settings, masker, chips_slot, step_slots, mapping_slot)
    if answer is None:
        return
    step_slots[3].markdown(
        step_card("4", "Cloud model response", answer.masked_answer, done=True),
        unsafe_allow_html=True,
    )
    result_slot.markdown(
        f"<div class='result'>{escape(answer.answer)}</div>", unsafe_allow_html=True
    )


def render_local(result, settings, masker, chips_slot, step_slots, mapping_slot) -> None:
    """Fill the panels that describe work already done on this machine."""
    chips_slot.markdown(detected_chips(result, settings), unsafe_allow_html=True)
    with mapping_slot.expander("local vault -- never leaves this machine"):
        st.code(vault_mapping(masker, result), language=None)

    step_slots[0].markdown(
        step_card("1", "PII detection", detection_summary(result), done=True),
        unsafe_allow_html=True,
    )
    step_slots[1].markdown(
        step_card("2", "Local pseudonymization", vault_summary(masker), done=True),
        unsafe_allow_html=True,
    )
    step_slots[2].markdown(
        step_card("3", "Cloud payload (anonymized)", result.masked_text, done=True),
        unsafe_allow_html=True,
    )


def send_to_cloud(result, settings, masker, step_slots, result_slot):
    """Everything from here on crosses the network.

    Returns the answer, or None if nothing came back. The caller keeps it:
    Streamlit reruns the whole script on the next click, and an answer that
    lives only in this call would vanish with it.
    """
    leaked = masker.leaks(result.masked_text)
    if leaked:
        step_slots[3].markdown(
            step_card("4", "Blocked", f"{len(leaked)} value(s) survived masking. Nothing sent.",
                      done=False),
            unsafe_allow_html=True,
        )
        return None

    if not os.getenv(settings["key_variable"]):
        step_slots[3].markdown(
            step_card(
                "4",
                "Cloud model response",
                f"{settings['key_variable']} is not set, so nothing was sent.\n"
                "Everything above ran locally. Put the key in .env to finish the round trip.",
                done=False,
            ),
            unsafe_allow_html=True,
        )
        result_slot.markdown(
            "<div class='result idle'>no API key set &mdash;<br>"
            "the local half of the pipeline still ran</div>",
            unsafe_allow_html=True,
        )
        return None

    # Written before the call, so the wait has something to show for itself.
    step_slots[3].markdown(
        step_card("4", "Cloud model response", "contacting the model...", done=False),
        unsafe_allow_html=True,
    )
    result_slot.markdown(
        "<div class='result idle'>waiting for the model&hellip;</div>",
        unsafe_allow_html=True,
    )

    pipeline = PrivacyPipeline(build_llm(settings["provider"], settings["model"]), masker)
    try:
        with st.spinner("Waiting for the cloud model"):
            answer = pipeline.ask_masked(result)
    except (ModelUnavailable, LeakDetected) as error:
        step_slots[3].markdown(
            step_card("4", "Cloud model response", f"Failed: {error}", done=False),
            unsafe_allow_html=True,
        )
        result_slot.markdown(
            "<div class='result idle'>nothing came back</div>", unsafe_allow_html=True
        )
        return None

    step_slots[3].markdown(
        step_card("4", "Cloud model response", answer.masked_answer, done=True),
        unsafe_allow_html=True,
    )
    result_slot.markdown(
        f"<div class='result'>{escape(answer.answer)}</div>", unsafe_allow_html=True
    )
    return answer


def session_masker(settings) -> Masker:
    """One masker per session, so placeholders stay stable between questions."""
    signature = (
        settings["threshold"],
        settings["mask_money"],
        settings["mask_malformed"],
        settings["match"],
    )
    if st.session_state.get("signature") != signature:
        st.session_state["signature"] = signature
        st.session_state["masker"] = Masker(
            detector=PiiDetector(
                score_threshold=settings["threshold"],
                detect_money=settings["mask_money"],
                mask_malformed=settings["mask_malformed"],
            ),
            vault=PseudonymVault(match=settings["match"]),
        )
    return st.session_state["masker"]


@st.cache_resource(show_spinner=False)
def survey_detector(threshold: float) -> PiiDetector:
    """Finds everything, including what we choose not to mask.

    Amounts and dates are usually what the model needs in order to reason at
    all, so they are shown as found-and-left rather than quietly ignored.
    """
    return PiiDetector(score_threshold=threshold, detect_money=True)


def detected_chips(result, settings) -> str:
    if result is None:
        return (
            "<div class='note'>detected before processing</div>"
            "<div class='chips'><span class='chip none'>nothing scanned yet</span></div>"
        )

    masked = sorted(result.entity_counts)
    found = survey_detector(settings["threshold"]).detect(result.original_text)
    left = sorted({e.entity_type for e in found} - set(masked))

    chips = "".join(f"<span class='chip'>{short(t)}</span>" for t in masked)
    chips += "".join(f"<span class='chip clear'>{short(t)}</span>" for t in left)
    if not chips:
        chips = "<span class='chip none'>nothing sensitive found</span>"

    note = "<div class='note'>detected and masked</div>"
    if left:
        note = (
            "<div class='note'>detected and masked &middot; "
            "<span class='clear'>left in the clear on purpose</span></div>"
        )
    return f"{note}<div class='chips'>{chips}</div>"


def detection_summary(result) -> str:
    """What was found, described without being disclosed.

    This card is on the projector. It names types, confidence and the token
    each span became -- never the value itself, which is already visible in
    the local column where it belongs.
    """
    if not result.entities:
        return "nothing sensitive found"
    lines = [
        f"{e.entity_type:<16} {e.score:.2f}  ->  {p.placeholder}"
        for e, p in zip(result.entities, result.pseudonyms)
    ]
    if result.sealed:
        lines.append(f"+ {len(result.sealed)} later mention(s) sealed by the sweep")
    return "\n".join(lines)


def vault_summary(masker: Masker) -> str:
    """The tokens minted, counted by type. The mapping itself stays local."""
    entries = masker.vault.entries()
    if not entries:
        return "nothing to pseudonymize"
    lines = [f"{p.placeholder}" for p in entries]
    return (
        f"{len(entries)} value(s) held locally, never sent:\n" + "  ".join(lines)
    )


def vault_mapping(masker: Masker, result=None) -> str:
    """The real mapping, for the local column only.

    Entries carried over from an earlier submission are marked, because a name
    that is no longer in the text still sits in the vault and it is confusing
    to see it listed as though it had just been found.
    """
    entries = masker.vault.entries()
    if not entries:
        return "empty"
    current = {p.placeholder for p in (result.pseudonyms if result else [])}
    return "\n".join(
        f"{p.placeholder}  <-  {shorten(p.original)}"
        + ("" if p.placeholder in current else "   (from an earlier submission)")
        for p in entries
    )


def step_card(number: str, title: str, body: str, *, done: bool) -> str:
    state = " done" if done else ""
    idle = "" if done else " idle"
    return (
        f"<div class='card{state}'><span class='n'>{number}</span>"
        f"<span class='h'>{title}</span>"
        f"<div class='body{idle}'>{escape(body)}</div></div>"
    )


def column_head(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='colhead'><div class='t'>{title}</div>"
        f"<div class='s'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def topbar(settings) -> None:
    st.markdown(
        "<div class='topbar'>"
        "<div class='brand'><span class='mark'>&#9678;</span>AnonAgent</div>"
        "<div class='status'><span class='dot'></span>local guard active"
        f"&nbsp;&nbsp;&middot;&nbsp;&nbsp;{settings['provider']} / {settings['model']}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def sidebar() -> dict:
    with st.sidebar:
        st.caption("Settings")
        provider = st.selectbox("Provider", ["openai", "groq", "ollama"], index=0)
        model = st.text_input("Model", value=DEFAULT_MODELS[provider])
        threshold = st.slider("Detection threshold", 0.1, 0.95, 0.4, 0.05)
        mask_money = st.checkbox(
            "Mask amounts too",
            value=False,
            help="Off by default: hiding the numbers is what stops the model "
            "from reasoning about them.",
        )
        mask_malformed = st.checkbox(
            "Mask malformed identifiers",
            value=False,
            help="Off: only account and identity numbers that pass their "
            "checksum are masked. On: the shape alone is enough, so a "
            "mistyped IBAN is covered too.",
        )
        match = st.radio("Matching", ["normalized", "exact"], horizontal=True)
        st.caption("A fresh vault starts whenever these change.")
    return {
        "provider": provider,
        "model": model,
        "threshold": threshold,
        "mask_money": mask_money,
        "mask_malformed": mask_malformed,
        "match": match,
        "key_variable": {"openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}.get(
            provider, "OLLAMA_BASE_URL"
        ),
    }


def shorten(value: str, limit: int = 42) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def short(entity_type: str) -> str:
    return entity_type.replace("_ADDRESS", "").replace("_CODE", "").replace("_NUMBER", "")


def escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


main()
