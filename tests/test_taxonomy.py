import pytest
from needlestack import taxonomy
from needlestack.taxonomy import RAILROAD, NAVAL, get_domain, DOMAINS


# --- backward-compatible module-level helpers (RAILROAD wrappers) ---

def test_synonyms_for_known_term():
    syn = taxonomy.synonyms_for("caboose")
    assert "waycar" in syn and "crummy" in syn
    assert "caboose" not in syn  # input itself excluded


def test_synonyms_for_via_synonym_input():
    """Looking up by a synonym returns the canonical name and sibling synonyms."""
    syn = taxonomy.synonyms_for("reefer")
    assert "refrigerator car" in syn
    assert "reefer" not in syn


def test_synonyms_for_is_case_insensitive():
    assert taxonomy.synonyms_for("CABOOSE") == taxonomy.synonyms_for("caboose")


def test_synonyms_for_unknown_term_empty():
    assert taxonomy.synonyms_for("automobile") == []


def test_synonyms_for_whitespace():
    assert taxonomy.synonyms_for("  caboose  ") == taxonomy.synonyms_for("caboose")


# --- single-source drift guards ---

def test_frequency_terms_derive_from_equipment():
    """Doctor's coverage chart must consume the same source as validation —
    no separate hand-maintained list that can drift."""
    assert set(taxonomy.frequency_terms()) == taxonomy.valid_equipment_types()


def test_valid_types_are_canonical_keys():
    assert taxonomy.valid_equipment_types() == set(taxonomy.EQUIPMENT.keys())


def test_prompt_helpers_nonempty():
    assert "caboose" in taxonomy.equipment_terms_prompt()
    assert taxonomy.settings_prompt()
    assert "Northern" in taxonomy.wheel_arrangements_prompt()


# --- Domain dataclass ---

def test_railroad_domain_fields():
    assert RAILROAD.name == "railroad"
    assert RAILROAD.subject_field == "is_railroad"
    assert RAILROAD.items_field == "equipment"
    assert RAILROAD.identifier_label == "reporting marks"
    assert "steam locomotive" in RAILROAD.valid_subject_types


def test_naval_domain_fields():
    assert NAVAL.name == "naval"
    assert NAVAL.subject_field == "is_naval"
    assert NAVAL.items_field == "vessels"
    assert NAVAL.identifier_label == "hull number"
    assert "destroyer" in NAVAL.valid_subject_types


def test_domain_item_fields_have_type_and_details():
    """Every domain must start with type (mid) and end with details (phrase)."""
    for domain in DOMAINS.values():
        field_names = [f for f, _ in domain.item_fields]
        weights = {f: w for f, w in domain.item_fields}
        assert field_names[0] == "type", f"{domain.name}: first item_field must be 'type'"
        assert "details" in field_names, f"{domain.name}: 'details' must be in item_fields"
        assert weights["type"] == "mid", f"{domain.name}: 'type' must be mid-priority"
        assert weights["details"] == "phrase", f"{domain.name}: 'details' must be phrase"


def test_domain_prompt_fragments_required_keys():
    required = {"preamble", "subject_qualifier", "item_singular", "id_instruction",
                "era_examples", "view_instruction", "fallback_preamble"}
    for domain in DOMAINS.values():
        missing = required - set(domain.prompt_fragments)
        assert not missing, f"{domain.name} missing prompt_fragments: {missing}"


def test_naval_synonyms_for_known_term():
    syn = NAVAL.synonyms_for("destroyer")
    assert "DD" in syn
    assert "tin can" in syn
    assert "destroyer" not in syn


def test_naval_synonyms_for_via_abbreviation():
    syn = NAVAL.synonyms_for("DD")
    assert "destroyer" in syn
    assert "tin can" in syn


def test_naval_synonyms_for_unknown():
    assert NAVAL.synonyms_for("locomotive") == []


def test_domain_frequency_terms_match_valid_types():
    """No drift between frequency_terms and valid_subject_types for any domain."""
    for domain in DOMAINS.values():
        assert set(domain.frequency_terms()) == domain.valid_subject_types, \
            f"Drift in {domain.name} domain"


# --- domain registry ---

def test_get_domain_railroad():
    assert get_domain("railroad") is RAILROAD


def test_get_domain_naval():
    assert get_domain("naval") is NAVAL


def test_get_domain_unknown_raises():
    with pytest.raises(KeyError, match="Unknown domain"):
        get_domain("aviation")


def test_domains_registry_contains_known_domains():
    assert "railroad" in DOMAINS
    assert "naval" in DOMAINS


def test_domains_are_independent():
    """The two domains must not share field names that would cause cross-domain confusion."""
    assert RAILROAD.subject_field != NAVAL.subject_field
    assert RAILROAD.items_field != NAVAL.items_field
