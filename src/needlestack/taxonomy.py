"""Single source of truth for photo collection domain vocabulary.

Each Domain bundles the vocabulary, JSON schema field names, and prompt fragments
needed by the captioner, query expander, and doctor. Nothing re-encodes domain rules
outside this module. (Syntactic-Semantic Seam rule: one source of truth.)
"""

from dataclasses import dataclass, field


@dataclass
class Domain:
    """Vocabulary and schema configuration for a photo collection domain.

    item_fields is an ordered list of (field_name, fts_weight) pairs that define
    every field in each subject item returned by the model:
      "high"   → mark_tokens (high FTS weight; identifiers like marks, hull numbers)
      "mid"    → equip_tokens (mid FTS weight; type names, road/class names)
      "phrase" → included in the human-readable caption phrase only
    "type" must always be first and "mid"; "details" should be last and "phrase".
    """
    name: str
    subject_types: dict[str, list[str]]  # canonical type → synonyms
    identifier_label: str                 # display: "reporting marks" / "hull number"
    settings: list[str]
    # JSON field names (vary so the model uses natural terminology per domain)
    subject_field: str    # boolean: "is_railroad" / "is_naval"
    items_field: str      # array: "equipment" / "vessels"
    # Per-item field definitions — order determines caption phrase word order
    item_fields: list[tuple[str, str]]
    # Prompt text fragments consumed by captioner._make_prompt()
    prompt_fragments: dict[str, str]
    # Optional notation dict (e.g. Whyte wheel arrangements for steam)
    notation: dict[str, str] = field(default_factory=dict)

    @property
    def valid_subject_types(self) -> frozenset[str]:
        return frozenset(self.subject_types)

    def synonyms_for(self, term: str) -> list[str]:
        """Alternate names for term (canonical or synonym), excluding the input itself."""
        t = term.strip().lower()
        out: list[str] = []
        for canonical, synonyms in self.subject_types.items():
            family = [canonical, *synonyms]
            if t in [f.lower() for f in family]:
                out.extend(f for f in family if f.lower() != t)
        return out

    def frequency_terms(self) -> list[str]:
        return list(self.subject_types)

    def subject_types_prompt(self) -> str:
        return ", ".join(self.subject_types)

    def settings_prompt(self) -> str:
        return ", ".join(self.settings)


# ---------------------------------------------------------------------------
# Railroad domain
# ---------------------------------------------------------------------------

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

SETTINGS: list[str] = [
    "yard", "depot", "station", "mainline", "siding", "branch line",
    "bridge", "trestle", "tunnel", "roundhouse", "turntable", "engine house",
    "water tower", "coaling tower", "signal", "interlocking tower",
    "grade crossing", "industrial spur",
]

RAILROAD = Domain(
    name="railroad",
    subject_types=EQUIPMENT,
    identifier_label="reporting marks",
    settings=SETTINGS,
    subject_field="is_railroad",
    items_field="equipment",
    item_fields=[
        ("type", "mid"),
        ("road_name", "mid"),
        ("reporting_marks", "high"),
        ("road_number", "high"),
        ("details", "phrase"),
    ],
    prompt_fragments={
        "preamble": "This is a railroad or railway photograph (or might be).",
        "subject_qualifier": "true only if the photo actually shows railroad subject matter",
        "item_singular": "piece of rolling stock",
        "id_instruction": (
            "Fill road_name (railroad company name), "
            "reporting_marks (e.g. ATSF, UP, SP), and road_number ONLY from text "
            "you can actually read on the equipment; leave blank if not legible."
        ),
        "era_examples": "steam era, 1950s diesel transition, modern",
        "view_instruction": (
            "broadside, three-quarter front, three-quarter rear, roster shot, "
            "action/moving, detail closeup, overhead, aerial"
        ),
        "type_note": (
            "For steam locomotives, if the wheel configuration is clearly legible, "
            "give the Whyte notation (e.g. 2-8-2 Mikado, 4-8-4 Northern)."
        ),
        "fallback_preamble": (
            "This is a railroad or railway photograph. Describe it for a searchable "
            "photo index using exact railroad terminology for any rolling stock, "
            "including railroad names, reporting marks, and road numbers if legible. "
            "Describe the setting and era. If this is not a railroad photo, describe "
            "what it actually shows with equal specificity. Write in plain sentences, "
            "no preamble."
        ),
    },
    notation=WHEEL_ARRANGEMENTS,
)


# ---------------------------------------------------------------------------
# Naval domain
# ---------------------------------------------------------------------------

_NAVAL_SUBJECT_TYPES: dict[str, list[str]] = {
    "battleship": ["BB", "dreadnought", "battlewagon"],
    "aircraft carrier": [
        "carrier", "flattop", "CV", "CVN", "CVA", "CVL", "CVE", "escort carrier",
    ],
    "destroyer": ["DD", "tin can"],
    "destroyer escort": ["DE"],
    "frigate": ["FF", "FFG"],
    "cruiser": [
        "CA", "CL", "CG", "CGN", "CLG",
        "heavy cruiser", "light cruiser", "guided missile cruiser",
    ],
    "submarine": ["SS", "SSN", "SSBN", "SSGN", "SSK", "sub", "boat", "pig boat"],
    "amphibious ship": ["LPH", "LSD", "LST", "LHA", "LPD", "LCC", "amphib"],
    "oiler": ["AO", "AOE", "AOR", "replenishment ship", "fleet oiler", "UNREP ship"],
    "destroyer tender": ["AD"],
    "submarine tender": ["AS"],
    "minesweeper": ["AM", "MSC", "MSO"],
    "patrol craft": ["PC", "PG", "PCF", "gunboat", "PT boat"],
    "landing craft": ["LCM", "LCVP", "LCU", "Higgins boat"],
    "tugboat": ["ATF", "YTB", "harbor tug"],
    "hospital ship": ["AH"],
}

_NAVAL_SETTINGS: list[str] = [
    "underway", "alongside pier", "at anchor", "anchorage",
    "drydock", "dry dock", "navy yard", "shipyard", "naval station",
    "sea trial", "fleet review", "fleet week",
]

NAVAL = Domain(
    name="naval",
    subject_types=_NAVAL_SUBJECT_TYPES,
    identifier_label="hull number",
    settings=_NAVAL_SETTINGS,
    subject_field="is_naval",
    items_field="vessels",
    item_fields=[
        ("type", "mid"),
        ("class_name", "mid"),
        ("hull_number", "high"),
        ("ship_name", "high"),
        ("details", "phrase"),
    ],
    prompt_fragments={
        "preamble": "This is a naval or maritime photograph (or might be).",
        "subject_qualifier": (
            "true only if the photo actually shows naval or military maritime subject matter"
        ),
        "item_singular": "vessel",
        "id_instruction": (
            "Fill hull_number (e.g. DD-963, CVN-65, SSN-571) and ship_name "
            "(e.g. USS Enterprise, USS Missouri) ONLY from markings or text you can "
            "actually read in the image; leave blank if not legible. "
            "Fill class_name (e.g. Iowa-class, Spruance-class) only when you are certain."
        ),
        "era_examples": "pre-war, WWII, Cold War, Vietnam era, modern",
        "view_instruction": (
            "broadside, bow quarter, stern quarter, aerial/overhead, "
            "drydock, detail closeup"
        ),
        "type_note": "",
        "fallback_preamble": (
            "This is a naval or maritime photograph. Describe it for a searchable "
            "photo index using exact naval terminology — ship type, class, hull number, "
            "ship name if legible, and setting. If this is not a naval photo, describe "
            "what it actually shows with equal specificity. Write in plain sentences, "
            "no preamble."
        ),
    },
)


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAINS: dict[str, Domain] = {
    "railroad": RAILROAD,
    "naval": NAVAL,
}


def get_domain(name: str) -> Domain:
    """Return a Domain by name. Raises KeyError for unknown names."""
    try:
        return DOMAINS[name]
    except KeyError:
        raise KeyError(
            f"Unknown domain {name!r}. Available: {', '.join(DOMAINS)}"
        ) from None


# ---------------------------------------------------------------------------
# Backward-compatible module-level helpers (delegate to RAILROAD)
# ---------------------------------------------------------------------------

def equipment_terms_prompt() -> str:
    """Comma-separated canonical railroad equipment terms."""
    return RAILROAD.subject_types_prompt()


def settings_prompt() -> str:
    """Comma-separated railroad setting terms."""
    return RAILROAD.settings_prompt()


def wheel_arrangements_prompt() -> str:
    """Whyte notation -> name pairs, for injecting into a prompt."""
    return ", ".join(
        f"{whyte} ({name})" for whyte, name in WHEEL_ARRANGEMENTS.items()
    )


def synonyms_for(term: str) -> list[str]:
    """Railroad synonym lookup. Delegates to RAILROAD.synonyms_for()."""
    return RAILROAD.synonyms_for(term)


VALID_EQUIPMENT_TYPES = frozenset(EQUIPMENT)


def valid_equipment_types() -> frozenset[str]:
    """Canonical railroad equipment types. Delegates to RAILROAD."""
    return RAILROAD.valid_subject_types


def frequency_terms() -> list[str]:
    """Terms the doctor coverage bar chart iterates over (railroad)."""
    return RAILROAD.frequency_terms()
