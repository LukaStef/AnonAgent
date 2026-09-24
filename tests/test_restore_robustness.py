"""Restoring placeholders the model gave back in its own spelling.

The system prompt asks for tokens echoed verbatim. Models comply most of the
time. The rest of the time the user must still get their data back, because
a placeholder left unresolved is a visible failure in the final answer.
"""

import pytest

from anonagent.privacy.vault import PseudonymVault


@pytest.fixture
def vault():
    v = PseudonymVault()
    v.pseudonymize("James Bond", "PERSON")
    v.pseudonymize("james@example.com", "EMAIL_ADDRESS")
    return v


@pytest.mark.parametrize(
    "mangled",
    [
        "<PERSON_001>",
        "PERSON_001",
        "**<PERSON_001>**",
        "`PERSON_001`",
        "[PERSON_001]",
        "(PERSON_001)",
        '"PERSON_001"',
        "PERSON 001",
        "PERSON-001",
        "Person_001",
        "person_001",
        "PERSON_1",
        "PERSON_0001",
    ],
)
def test_a_mangled_placeholder_still_resolves(vault, mangled):
    assert "James Bond" in vault.restore(mangled)


def test_a_multi_word_entity_type_resolves(vault):
    assert vault.restore("Email_Address_1") == "james@example.com"
    assert vault.restore("<EMAIL_ADDRESS_001>") == "james@example.com"


def test_punctuation_the_model_added_is_kept(vault):
    """Angle brackets are ours to remove. Everything else is the model's prose."""
    assert vault.restore("[PERSON_001]") == "[James Bond]"
    assert vault.restore("(PERSON_001)") == "(James Bond)"
    assert vault.restore("**PERSON_001**") == "**James Bond**"


def test_possessives_and_surrounding_words_survive(vault):
    assert vault.restore("the user PERSON_001's account") == (
        "the user James Bond's account"
    )


@pytest.mark.parametrize(
    "innocent",
    [
        "COVID 19",
        "section 3 of the report",
        "Figure 2 shows the trend",
        "ISO 27001",
        "PERSON_099",
        "EMAIL_ADDRESS_042",
    ],
)
def test_text_that_only_looks_like_a_placeholder_is_left_alone(vault, innocent):
    """Being liberal must not mean rewriting ordinary prose."""
    assert vault.restore(innocent) == innocent


def test_an_empty_vault_changes_nothing():
    assert PseudonymVault().restore("PERSON_001 and PERSON_002") == (
        "PERSON_001 and PERSON_002"
    )


def test_several_mangled_placeholders_in_one_sentence(vault):
    reply = "Person 1 can be reached at email_address_001, per <PERSON_001>."
    assert vault.restore(reply) == (
        "James Bond can be reached at james@example.com, "
        "per James Bond."
    )


def test_restoration_does_not_run_twice_over_its_own_output():
    """A restored value that looks like a placeholder must not be re-restored."""
    vault = PseudonymVault()
    vault.pseudonymize("ORDER_002", "PERSON")
    vault.pseudonymize("Maria", "PERSON")

    assert vault.restore("<PERSON_001>") == "ORDER_002"
