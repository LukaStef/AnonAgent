"""How the vault decides that two mentions are the same value.

These lock in the trade-off between the two matching modes: "normalized"
collapses spellings and restores canonically, "exact" keeps them apart and
restores byte-for-byte.
"""

import pytest

from anonagent.privacy.vault import PseudonymVault

SPELLINGS = ["nikola@example.com", "Nikola@Example.COM", "NIKOLA@EXAMPLE.COM"]


def test_normalized_mode_collapses_spellings_into_one_placeholder():
    vault = PseudonymVault(match="normalized")
    placeholders = {vault.pseudonymize(s, "EMAIL_ADDRESS").placeholder for s in SPELLINGS}
    assert placeholders == {"<EMAIL_ADDRESS_001>"}
    assert len(vault) == 1


def test_normalized_mode_restores_every_spelling_as_the_first_one():
    vault = PseudonymVault(match="normalized")
    for spelling in SPELLINGS:
        vault.pseudonymize(spelling, "EMAIL_ADDRESS")
    assert vault.restore("<EMAIL_ADDRESS_001>") == "nikola@example.com"


def test_exact_mode_keeps_spellings_apart():
    vault = PseudonymVault(match="exact")
    placeholders = [vault.pseudonymize(s, "EMAIL_ADDRESS").placeholder for s in SPELLINGS]
    assert placeholders == ["<EMAIL_ADDRESS_001>", "<EMAIL_ADDRESS_002>", "<EMAIL_ADDRESS_003>"]


def test_exact_mode_restores_each_spelling_untouched():
    vault = PseudonymVault(match="exact")
    text = " and ".join(vault.pseudonymize(s, "EMAIL_ADDRESS").placeholder for s in SPELLINGS)
    assert vault.restore(text) == " and ".join(SPELLINGS)


def test_truly_different_values_never_collapse():
    vault = PseudonymVault(match="normalized")
    mine = vault.pseudonymize("nikola@example.com", "EMAIL_ADDRESS")
    theirs = vault.pseudonymize("team@example.com", "EMAIL_ADDRESS")
    assert mine.placeholder != theirs.placeholder


def test_spellings_fingerprint_alike_in_both_modes():
    """The link between spellings survives even when placeholders differ."""
    for match in ("normalized", "exact"):
        vault = PseudonymVault(secret="shared", match=match)
        fingerprints = {vault.pseudonymize(s, "EMAIL_ADDRESS").fingerprint for s in SPELLINGS}
        assert len(fingerprints) == 1, match


def test_variants_records_each_spelling_once():
    vault = PseudonymVault(match="normalized")
    for spelling in SPELLINGS + [SPELLINGS[0]]:
        vault.pseudonymize(spelling, "EMAIL_ADDRESS")
    assert vault.variants("<EMAIL_ADDRESS_001>") == SPELLINGS


def test_canonicalized_reports_only_values_that_changed_spelling():
    vault = PseudonymVault(match="normalized")
    vault.pseudonymize("nikola@example.com", "EMAIL_ADDRESS")
    vault.pseudonymize("NIKOLA@EXAMPLE.COM", "EMAIL_ADDRESS")
    vault.pseudonymize("team@example.com", "EMAIL_ADDRESS")

    assert [p.placeholder for p in vault.canonicalized()] == ["<EMAIL_ADDRESS_001>"]


def test_nothing_is_canonicalized_in_exact_mode():
    vault = PseudonymVault(match="exact")
    for spelling in SPELLINGS:
        vault.pseudonymize(spelling, "EMAIL_ADDRESS")
    assert vault.canonicalized() == []


def test_an_unknown_match_mode_is_rejected():
    with pytest.raises(ValueError, match="normalized"):
        PseudonymVault(match="fuzzy")
