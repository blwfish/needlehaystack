"""Single source of truth for railroad vocabulary.

Every site that needs railroad terminology — the caption prompt, the query-expansion
prompt, the query-expansion local synonym lookup, and the doctor frequency chart —
consumes this module. Nothing re-encodes the vocabulary in its own copy. (Companion to
the Syntactic-Semantic Seam rule: a semantic distinction gets one source of truth.)
"""

# Canonical equipment type -> alternate names / synonyms for the SAME thing.
# Synonyms are strictly other names for the identical subject, never related items —
# the query expander relies on that to widen recall without drifting topic.
EQUIPMENT: dict[str, list[str]] = {
    # Motive power
    "steam locomotive": ["steam engine", "steamer", "steam loco"],
    "diesel locomotive": ["diesel engine", "diesel loco", "diesel unit"],
    "electric locomotive": ["electric engine", "electric loco", "motor"],
    # Rolling stock
    "boxcar": ["box car", "boxcars", "house car"],
    "flatcar": ["flat car", "flatcars"],
    "gondola": ["gondola car", "gon"],
    "hopper car": ["hopper", "hoppers", "coal hopper", "ore car"],
    "tank car": ["tanker", "cistern car", "pressure car", "tank wagon"],
    "refrigerator car": ["reefer", "refrigerated car", "ice car"],
    "stock car": ["cattle car", "livestock car"],
    "well car": ["intermodal car", "double stack car", "container car"],
    "auto rack": ["autorack", "auto carrier", "car carrier"],
    "passenger coach": ["coach", "passenger car", "day coach", "rider car"],
    "baggage car": ["baggage", "express car"],
    "observation car": ["obs car", "observation"],
    "dome car": ["vista dome", "dome"],
    "caboose": ["cabin car", "waycar", "way car", "hack", "crummy", "van", "bobber"],
    # Maintenance-of-way / non-revenue
    "tender": ["coal tender", "water tender"],
    "snowplow": ["snow plow", "rotary plow", "russell plow"],
    "crane": ["wrecker", "derrick", "big hook"],
}

# Common Whyte-notation steam wheel arrangements, included in the caption prompt so the
# model names them when legible (e.g. "4-8-4 Northern").
WHEEL_ARRANGEMENTS: dict[str, str] = {
    "4-4-0": "American",
    "2-6-0": "Mogul",
    "2-8-0": "Consolidation",
    "4-6-2": "Pacific",
    "2-8-2": "Mikado",
    "4-8-2": "Mountain",
    "4-8-4": "Northern",
    "2-10-2": "Santa Fe",
    "4-6-4": "Hudson",
    "2-8-4": "Berkshire",
}

# Settings / infrastructure the caption prompt asks the model to identify.
SETTINGS: list[str] = [
    "yard", "depot", "station", "mainline", "siding", "branch line",
    "bridge", "trestle", "tunnel", "roundhouse", "turntable", "engine house",
    "water tower", "coaling tower", "signal", "interlocking tower",
    "grade crossing", "industrial spur",
]


def equipment_terms_prompt() -> str:
    """Comma-separated canonical equipment terms, for injecting into a model prompt."""
    return ", ".join(EQUIPMENT.keys())


def settings_prompt() -> str:
    """Comma-separated setting terms, for injecting into a model prompt."""
    return ", ".join(SETTINGS)


def wheel_arrangements_prompt() -> str:
    """Whyte notation -> name pairs, for injecting into a model prompt."""
    return ", ".join(f"{whyte} ({name})" for whyte, name in WHEEL_ARRANGEMENTS.items())


def synonyms_for(term: str) -> list[str]:
    """Deterministic local synonym lookup, case-insensitive.

    Returns alternate names for ``term`` (canonical or synonym), excluding the input
    itself. Empty list when the term is unknown — the caller falls back to LLM
    expansion for those.
    """
    t = term.strip().lower()
    out: list[str] = []
    for canonical, synonyms in EQUIPMENT.items():
        family = [canonical, *synonyms]
        family_lower = [f.lower() for f in family]
        if t in family_lower:
            out.extend(f for f, fl in zip(family, family_lower) if fl != t)
    return out


VALID_EQUIPMENT_TYPES = frozenset(EQUIPMENT)


def valid_equipment_types() -> frozenset[str]:
    """Canonical equipment types, for validating model-emitted categoricals.
    Cached module-level — called once per captioned image."""
    return VALID_EQUIPMENT_TYPES


def frequency_terms() -> list[str]:
    """Terms the doctor coverage bar chart iterates over."""
    return list(EQUIPMENT.keys())
