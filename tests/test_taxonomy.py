import pytest
from needlestack import taxonomy
from needlestack.taxonomy import RAILROAD, NAVAL, ARMOR, AVIATION, BIRDS, get_domain, DOMAINS


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


def test_armor_domain_fields():
    assert ARMOR.name == "armor"
    assert ARMOR.subject_field == "is_armor"
    assert ARMOR.items_field == "vehicles"
    assert ARMOR.identifier_label == "tactical number"
    assert "tank" in ARMOR.valid_subject_types
    # Check high-priority identifier field exists
    weights = {f: w for f, w in ARMOR.item_fields}
    assert weights["tactical_number"] == "high"
    assert weights["vehicle_name"] == "mid"


def test_aviation_domain_fields():
    assert AVIATION.name == "aviation"
    assert AVIATION.subject_field == "is_aviation"
    assert AVIATION.items_field == "aircraft"
    assert AVIATION.identifier_label == "tail code"
    assert "fighter" in AVIATION.valid_subject_types
    weights = {f: w for f, w in AVIATION.item_fields}
    assert weights["tail_code"] == "high"
    assert weights["nickname"] == "high"
    assert weights["aircraft_model"] == "mid"


def test_domain_item_fields_have_type_and_details():
    """Every domain must start with type (mid), end with details (phrase), and have unique names."""
    for domain in DOMAINS.values():
        field_names = [f for f, _ in domain.item_fields]
        weights = {f: w for f, w in domain.item_fields}
        assert field_names[0] == "type", f"{domain.name}: first item_field must be 'type'"
        assert field_names[-1] == "details", f"{domain.name}: last item_field must be 'details'"
        assert weights["type"] == "mid", f"{domain.name}: 'type' must be mid-priority"
        assert weights["details"] == "phrase", f"{domain.name}: 'details' must be phrase"
        assert len(field_names) == len(set(field_names)), \
            f"{domain.name}: item_field names must be unique (duplicate would double-count in caption)"


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
        get_domain("spaceflight")


def test_domains_registry_contains_known_domains():
    assert "railroad" in DOMAINS
    assert "naval" in DOMAINS
    assert "armor" in DOMAINS
    assert "aviation" in DOMAINS
    assert "birds" in DOMAINS


def test_all_domain_subject_fields_are_unique():
    """Every domain uses a distinct subject_field name — cross-domain JSON ambiguity check."""
    subject_fields = [d.subject_field for d in DOMAINS.values()]
    assert len(subject_fields) == len(set(subject_fields)), \
        f"Duplicate subject_field: {subject_fields}"


def test_all_domain_items_fields_are_unique():
    """Every domain uses a distinct items_field name — no two domains share a JSON key."""
    items_fields = [d.items_field for d in DOMAINS.values()]
    assert len(items_fields) == len(set(items_fields)), \
        f"Duplicate items_field: {items_fields}"


def test_armor_synonyms_for_known_term():
    syn = ARMOR.synonyms_for("tank")
    assert "MBT" in syn
    assert "main battle tank" in syn
    assert "tank" not in syn


def test_aviation_synonyms_for_known_term():
    syn = AVIATION.synonyms_for("fighter")
    assert "interceptor" in syn
    assert "fighter" not in syn


def test_aviation_synonyms_for_abbreviation():
    syn = AVIATION.synonyms_for("UAV")
    assert "drone" in syn


# --- birds domain ---

def test_get_domain_birds():
    assert get_domain("birds") is BIRDS


def test_birds_domain_fields():
    assert BIRDS.name == "birds"
    assert BIRDS.subject_field == "is_birds"
    assert BIRDS.items_field == "birds"
    assert BIRDS.identifier_label == "species"
    assert "raptor" in BIRDS.valid_subject_types
    assert "waterfowl" in BIRDS.valid_subject_types
    assert "songbird" in BIRDS.valid_subject_types


def test_birds_domain_high_weight_fields():
    """common_name and species must be high-weight so species searches rank well."""
    weights = {f: w for f, w in BIRDS.item_fields}
    assert weights["common_name"] == "high"
    assert weights["species"] == "high"
    assert weights["behavior"] == "mid"
    assert weights["plumage"] == "mid"


def test_birds_synonyms_for_canonical_raptor():
    """Looking up the canonical type returns its synonyms."""
    syn = BIRDS.synonyms_for("raptor")
    assert "hawk" in syn
    assert "eagle" in syn
    assert "falcon" in syn
    assert "osprey" in syn
    assert "raptor" not in syn


def test_birds_synonyms_for_via_synonym():
    """Looking up a synonym returns the canonical name and sibling synonyms."""
    syn = BIRDS.synonyms_for("hawk")
    assert "raptor" in syn
    assert "eagle" in syn
    assert "hawk" not in syn


def test_birds_synonyms_for_heron():
    syn = BIRDS.synonyms_for("heron")
    assert "wading bird" in syn
    assert "egret" in syn
    assert "crane" in syn


def test_birds_synonyms_for_unknown():
    assert BIRDS.synonyms_for("locomotive") == []
