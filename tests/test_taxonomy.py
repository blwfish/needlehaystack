from needlestack import taxonomy


# --- synonyms_for ---

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
