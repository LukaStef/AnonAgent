from anonagent.privacy.vault import PseudonymVault


def test_same_value_gets_same_placeholder():
    vault = PseudonymVault()
    first = vault.pseudonymize("James Bond", "PERSON")
    second = vault.pseudonymize("James Bond", "PERSON")
    assert first.placeholder == second.placeholder
    assert len(vault) == 1


def test_matching_ignores_case_and_extra_whitespace():
    vault = PseudonymVault()
    first = vault.pseudonymize("James Bond", "PERSON")
    second = vault.pseudonymize("james   BOND", "PERSON")
    assert first.placeholder == second.placeholder


def test_distinct_values_get_distinct_placeholders():
    vault = PseudonymVault()
    first = vault.pseudonymize("James", "PERSON")
    second = vault.pseudonymize("Maria", "PERSON")
    assert first.placeholder != second.placeholder
    assert (first.placeholder, second.placeholder) == ("<PERSON_001>", "<PERSON_002>")


def test_counters_are_per_entity_type():
    vault = PseudonymVault()
    person = vault.pseudonymize("James", "PERSON")
    email = vault.pseudonymize("james@example.com", "EMAIL_ADDRESS")
    assert person.placeholder == "<PERSON_001>"
    assert email.placeholder == "<EMAIL_ADDRESS_001>"


def test_restore_tolerates_missing_angle_brackets():
    vault = PseudonymVault()
    vault.pseudonymize("James", "PERSON")
    assert vault.restore("PERSON_001 has funds") == "James has funds"
    assert vault.restore("<PERSON_001> has funds") == "James has funds"


def test_restore_leaves_unknown_placeholders_alone():
    vault = PseudonymVault()
    vault.pseudonymize("James", "PERSON")
    assert vault.restore("<PERSON_099> is unknown") == "<PERSON_099> is unknown"


def test_restore_is_a_noop_on_an_empty_vault():
    assert PseudonymVault().restore("nothing to do") == "nothing to do"


def test_fingerprints_are_stable_across_vaults_sharing_a_secret():
    one = PseudonymVault(secret="shared-secret")
    two = PseudonymVault(secret="shared-secret")
    assert (
        one.pseudonymize("James", "PERSON").fingerprint
        == two.pseudonymize("James", "PERSON").fingerprint
    )


def test_fingerprints_differ_under_different_secrets():
    one = PseudonymVault(secret="secret-a")
    two = PseudonymVault(secret="secret-b")
    assert (
        one.pseudonymize("James", "PERSON").fingerprint
        != two.pseudonymize("James", "PERSON").fingerprint
    )


def test_resolve_returns_the_original_value():
    vault = PseudonymVault()
    pseudonym = vault.pseudonymize("James", "PERSON")
    assert vault.resolve(pseudonym.placeholder).original == "James"
    assert vault.resolve("PERSON_001").original == "James"
    assert vault.resolve("<PERSON_404>") is None
