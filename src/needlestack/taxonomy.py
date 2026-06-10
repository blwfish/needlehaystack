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
# Armor domain (AFVs / armored fighting vehicles)
# ---------------------------------------------------------------------------

_ARMOR_SUBJECT_TYPES: dict[str, list[str]] = {
    "tank": [
        "MBT", "main battle tank", "medium tank", "heavy tank", "light tank",
        "infantry tank", "cavalry tank", "cruiser tank",
    ],
    "armored personnel carrier": ["APC", "troop carrier", "armored troop carrier"],
    "infantry fighting vehicle": ["IFV", "MICV", "mechanized infantry vehicle", "BMP", "Bradley"],
    "self-propelled gun": [
        "SPG", "SP gun", "self-propelled artillery", "assault gun",
        "tank destroyer", "SP howitzer", "SP mortar", "gun motor carriage",
    ],
    "half-track": ["halftrack", "semi-tracked vehicle", "armored halftrack"],
    "armored car": [
        "wheeled armored vehicle", "scout car", "armored reconnaissance vehicle",
        "armored fighting vehicle", "armored lorry",
    ],
    "armored recovery vehicle": ["ARV", "recovery vehicle", "BREM"],
    "combat engineer vehicle": ["CEV", "bridgelayer", "AVLB", "AVRE", "dozer tank"],
    "antiaircraft vehicle": [
        "SPAAG", "SPAA", "self-propelled AA", "AA vehicle", "Flakpanzer",
    ],
    "multiple launch rocket system": [
        "MLRS", "rocket artillery", "rocket launcher", "MRL", "Katyusha",
    ],
    "armored command vehicle": ["ACV", "command tank", "command vehicle"],
    "artillery tractor": ["prime mover", "gun tractor"],
}

_ARMOR_SETTINGS: list[str] = [
    "proving ground", "firing range", "field exercise", "maneuvers",
    "desert", "muddy terrain", "winter/snow", "urban",
    "museum display", "depot", "workshop", "parade", "static display",
]

ARMOR = Domain(
    name="armor",
    subject_types=_ARMOR_SUBJECT_TYPES,
    identifier_label="tactical number",
    settings=_ARMOR_SETTINGS,
    subject_field="is_armor",
    items_field="vehicles",
    item_fields=[
        ("type", "mid"),
        ("vehicle_name", "mid"),   # e.g. M4 Sherman, Tiger I, T-34/85
        ("nation", "mid"),          # e.g. Germany, United States, Soviet Union
        ("tactical_number", "high"), # painted hull/turret marking
        ("details", "phrase"),
    ],
    prompt_fragments={
        "preamble": "This is an armored fighting vehicle photograph (or might be).",
        "subject_qualifier": (
            "true only if the photo actually shows a tank, armored vehicle, "
            "or other AFV subject matter"
        ),
        "item_singular": "vehicle",
        "id_instruction": (
            "Fill vehicle_name (specific model, e.g. M4 Sherman, Tiger I, T-34/85) "
            "and nation (e.g. Germany, United States, Soviet Union, United Kingdom). "
            "Fill tactical_number (hull or turret marking, e.g. 121, B17) ONLY from "
            "markings you can actually read in the image; leave blank if not legible."
        ),
        "era_examples": "WWI, WWII, Cold War, Vietnam era, modern",
        "view_instruction": (
            "broadside, three-quarter front, three-quarter rear, head-on, rear, "
            "overhead, detail closeup, interior/fighting compartment"
        ),
        "type_note": "",
        "fallback_preamble": (
            "This is an armored fighting vehicle photograph. Describe it for a "
            "searchable photo index using exact AFV terminology — vehicle type, "
            "specific model name, nation, tactical markings if legible, and setting. "
            "If this is not an AFV photo, describe what it actually shows with equal "
            "specificity. Write in plain sentences, no preamble."
        ),
    },
)


# ---------------------------------------------------------------------------
# Aviation domain (military and historic aircraft)
# ---------------------------------------------------------------------------

_AVIATION_SUBJECT_TYPES: dict[str, list[str]] = {
    "fighter": [
        "pursuit aircraft", "interceptor", "air superiority fighter",
        "multirole fighter", "day fighter", "night fighter",
    ],
    "bomber": [
        "strategic bomber", "medium bomber", "light bomber", "dive bomber",
        "torpedo bomber", "flying fortress",
    ],
    "attack aircraft": [
        "close air support", "ground attack", "strike aircraft", "CAS aircraft",
    ],
    "transport": [
        "cargo aircraft", "airlift", "troop transport", "utility transport",
    ],
    "helicopter": [
        "rotorcraft", "chopper", "gunship", "attack helicopter",
        "utility helicopter", "observation helicopter",
    ],
    "trainer": [
        "training aircraft", "advanced trainer", "jet trainer", "basic trainer",
    ],
    "reconnaissance aircraft": [
        "recon", "photo-recon", "surveillance aircraft", "observation aircraft",
        "spyplane",
    ],
    "maritime patrol": [
        "ASW aircraft", "patrol aircraft", "flying boat", "anti-submarine",
        "maritime reconnaissance",
    ],
    "tanker": ["aerial refueling", "air-to-air refueling", "tanker aircraft"],
    "drone": ["UAV", "unmanned aerial vehicle", "remotely piloted aircraft", "UAS"],
    "glider": ["troop glider", "sailplane", "towed glider"],
    "seaplane": ["floatplane", "amphibian", "flying boat"],
}

_AVIATION_SETTINGS: list[str] = [
    "flight line", "ramp", "apron", "hangar", "in flight", "landing",
    "takeoff", "carrier deck", "museum", "airshow", "dispersal",
    "revetment", "airfield", "airstrip",
]

AVIATION = Domain(
    name="aviation",
    subject_types=_AVIATION_SUBJECT_TYPES,
    identifier_label="tail code",
    settings=_AVIATION_SETTINGS,
    subject_field="is_aviation",
    items_field="aircraft",
    item_fields=[
        ("type", "mid"),
        ("aircraft_model", "mid"),  # e.g. F-86 Sabre, B-17 Flying Fortress
        ("operator", "mid"),         # e.g. USAF, RAF, Luftwaffe
        ("tail_code", "high"),       # serial number or tail code
        ("nickname", "high"),        # e.g. Memphis Belle, Enola Gay
        ("details", "phrase"),
    ],
    prompt_fragments={
        "preamble": "This is an aviation photograph (or might be).",
        "subject_qualifier": (
            "true only if the photo actually shows aircraft subject matter"
        ),
        "item_singular": "aircraft",
        "id_instruction": (
            "Fill aircraft_model (specific type, e.g. F-86 Sabre, B-17 Flying Fortress, "
            "Spitfire Mk IX) and operator (e.g. USAF, RAF, US Navy, Luftwaffe). "
            "Fill tail_code (serial or buzz number, e.g. 44-83684, EE-549) and nickname "
            "(e.g. Memphis Belle, Enola Gay) ONLY from markings or text you can actually "
            "read in the image; leave blank if not legible."
        ),
        "era_examples": "WWI, interwar, WWII, Korean War, Cold War, Vietnam era, modern",
        "view_instruction": (
            "broadside/profile, three-quarter front, three-quarter rear, head-on, "
            "in flight, landing/takeoff, detail closeup, cockpit"
        ),
        "type_note": "",
        "fallback_preamble": (
            "This is an aviation photograph. Describe it for a searchable photo index "
            "using exact aviation terminology — aircraft type, specific model name, "
            "operator/service, tail code or serial if legible, and setting. "
            "If this is not an aviation photo, describe what it actually shows with "
            "equal specificity. Write in plain sentences, no preamble."
        ),
    },
)


# ---------------------------------------------------------------------------
# Birds domain (bird photography)
# ---------------------------------------------------------------------------

_BIRD_SUBJECT_TYPES: dict[str, list[str]] = {
    "raptor": [
        "hawk", "eagle", "falcon", "osprey", "kite", "harrier",
        "accipiter", "buteo", "buzzard", "vulture", "condor",
    ],
    "owl": [
        "screech owl", "great horned owl", "barn owl", "barred owl",
        "snowy owl", "burrowing owl", "short-eared owl", "long-eared owl",
    ],
    "waterfowl": [
        "duck", "goose", "swan", "teal", "pintail", "mallard",
        "merganser", "bufflehead", "goldeneye", "scoter", "eider",
        "shoveler", "wigeon", "canvasback", "redhead", "scaup",
    ],
    "shorebird": [
        "sandpiper", "plover", "godwit", "curlew", "dowitcher", "dunlin",
        "knot", "turnstone", "yellowlegs", "wader", "snipe", "phalarope",
        "oystercatcher", "avocet", "stilt",
    ],
    "wading bird": [
        "heron", "egret", "crane", "spoonbill", "ibis", "stork", "bittern",
    ],
    "songbird": [
        "passerine", "warbler", "sparrow", "finch", "thrush", "flycatcher",
        "vireo", "wren", "nuthatch", "chickadee", "titmouse", "tanager",
        "bunting", "oriole", "grosbeak", "bluebird", "robin", "catbird",
        "mockingbird", "starling", "blackbird", "meadowlark",
    ],
    "hummingbird": ["hummer"],
    "woodpecker": ["flicker", "sapsucker", "pileated woodpecker"],
    "seabird": [
        "gull", "tern", "pelican", "cormorant", "gannet", "albatross",
        "petrel", "shearwater", "booby", "frigatebird", "murre", "puffin",
        "auklet", "skua", "jaeger", "loon",
    ],
    "gamebird": [
        "grouse", "pheasant", "quail", "turkey", "ptarmigan",
        "prairie chicken", "partridge", "bobwhite",
    ],
    "corvid": ["crow", "raven", "jay", "magpie"],
    "dove": ["pigeon", "mourning dove", "collared dove", "rock pigeon", "band-tailed pigeon"],
    "kingfisher": [],
    "swallow": ["swift", "martin", "barn swallow", "cliff swallow", "tree swallow"],
    "rail": ["coot", "gallinule", "moorhen", "crake", "sora"],
}

_BIRD_SETTINGS: list[str] = [
    "wetland", "marsh", "pond", "lake", "river", "stream",
    "ocean", "pelagic", "coastal", "beach", "estuary", "mudflat", "tidal flat",
    "forest", "forest edge", "woodland",
    "grassland", "prairie", "scrub", "desert", "tundra", "alpine",
    "urban", "suburban", "backyard", "feeder",
    "agricultural field", "rocky coast", "cliffside",
]

BIRDS = Domain(
    name="birds",
    subject_types=_BIRD_SUBJECT_TYPES,
    identifier_label="species",
    settings=_BIRD_SETTINGS,
    subject_field="is_birds",
    items_field="birds",
    item_fields=[
        ("type", "mid"),           # bird group: raptor, waterfowl, songbird, etc.
        ("common_name", "high"),   # Red-tailed Hawk, Great Blue Heron
        ("species", "high"),       # Buteo jamaicensis — scientific name, optional
        ("behavior", "mid"),       # perched, in flight, foraging, displaying, preening
        ("plumage", "mid"),        # adult, juvenile, breeding, non-breeding, eclipse
        ("details", "phrase"),
    ],
    prompt_fragments={
        "preamble": "This is a bird photograph (or might be).",
        "subject_qualifier": "true only if the photo actually shows a bird or birds",
        "item_singular": "bird",
        "id_instruction": (
            "Identify each bird by common_name (e.g. Red-tailed Hawk, Great Blue Heron, "
            "Yellow Warbler) from its visual characteristics — plumage, size, shape, bill, "
            "and markings. Include species (scientific name, e.g. Buteo jamaicensis) if you "
            "are confident. If uncertain, give the most specific identification you can "
            "confidently make (e.g. 'Accipiter sp.' or 'Empidonax flycatcher'). "
            "Note any visible leg bands or color rings in details."
        ),
        "era_examples": "recent digital, 1990s film, 1980s, historic",
        "view_instruction": (
            "perched, in flight, landing, taking off, soaring, hovering, swimming, "
            "wading, diving, foraging, displaying, singing/calling, preening, at nest, "
            "with prey, flock"
        ),
        "type_note": "",
        "fallback_preamble": (
            "This is a bird photograph. Describe it for a searchable photo index "
            "using precise bird identification — common name, scientific name if known, "
            "plumage/age, behavior, and habitat. If the species is uncertain, give the "
            "most specific identification you can confidently make. "
            "If this is not a bird photo, describe what it actually shows with equal "
            "specificity. Write in plain sentences, no preamble."
        ),
    },
)


# ---------------------------------------------------------------------------
# Motorsports domain
# ---------------------------------------------------------------------------

_MOTORSPORTS_SUBJECT_TYPES: dict[str, list[str]] = {
    # Formula / open-wheel single-seaters
    "Formula 1 car": ["F1 car", "grand prix car", "Formula One car"],
    "Formula 2 car": ["F2 car", "Formula Two car"],
    "Formula 3 car": ["F3 car", "Formula Three car"],
    "Formula 4 car": ["F4 car"],
    "IndyCar": ["Indy car", "CART", "Champ Car", "championship car"],
    "Formula Ford": ["FF1600", "Formula Ford 1600"],
    "Formula Vee": ["FV", "Formula V"],
    "Formula Atlantic": ["Atlantic car"],
    # Sports prototypes
    "GTP prototype": ["GTP", "Hypercar", "LMH", "Le Mans Hypercar"],
    "LMP2 prototype": ["LMP2", "Le Mans Prototype 2"],
    "LMP3 prototype": ["LMP3"],
    "sports prototype": ["sports racer", "prototype", "P-class"],
    # GT / sports car
    "GT3 car": ["GT3"],
    "GT4 car": ["GT4"],
    "GTLM car": ["GTLM", "GT Le Mans"],
    "touring car": ["TCR", "TC", "TCA"],
    # Porsche Club Racing (highest priority)
    "Porsche 911 GT3 Cup": [
        "GT3 Cup car", "Cup car", "992 Cup", "991 Cup", "997 Cup", "996 Cup",
        "Porsche Cup car", "PCR Cup",
    ],
    "Porsche Cayman GT4 Clubsport": [
        "GT4 Clubsport", "Cayman GT4 CS", "982 Clubsport", "981 Clubsport",
        "Cayman Clubsport",
    ],
    "Porsche 718 Cayman": ["718 Cayman", "718 Club Sport"],
    "Porsche 911": ["911", "992", "991", "997", "996", "993", "964", "930"],
    # NASCAR
    "NASCAR Cup car": ["Cup Series car", "Next Gen car", "Cup car"],
    "NASCAR Xfinity car": ["Xfinity car", "Xfinity Series car"],
    "NASCAR Truck": ["Truck Series", "Craftsman Truck", "SuperTruck", "pickup race truck"],
    "Super Late Model": ["SLM", "full-bodied Late Model"],
    "Late Model": ["Late Model Stock", "LMS", "LMSC", "Late Model Stock Car"],
    "Limited Late Model": ["Limited Late Model Stock", "LLMS"],
    "Sportsman": ["Street Stock", "Hobby Stock", "Pure Stock", "Bombers"],
    "Modified": ["SK Modified", "Tour Modified", "Open Modified", "ACT Modified"],
    # IMSA / sports car
    "IMSA GTD Pro car": ["GTD Pro", "GTD Pro car"],
    "IMSA GTD car": ["GTD car", "GT Daytona"],
    # Vintage / historic
    "vintage race car": ["historic race car", "vintage racer", "historic racing car"],
    "vintage sports car": ["vintage sports racer", "vintage GT"],
    "vintage stock car": ["historic stock car"],
    "vintage open wheel": ["historic formula car", "vintage single-seater"],
    "vintage prototype": ["historic prototype", "vintage sports prototype"],
    # People / scene subjects (car_number, class, series left blank)
    "driver portrait": ["racing driver", "driver", "racer", "pilot", "helmeted driver"],
    "pit crew": ["crew member", "mechanic", "over-the-wall crew", "pit crew member"],
    "official": ["marshal", "flag marshal", "corner worker", "track official", "steward"],
    "spectator": ["fan", "crowd", "paddock visitor", "race fan"],
}

_MOTORSPORTS_SETTINGS: list[str] = [
    # On-track
    "on track", "straight", "start/finish straight", "back straight",
    "braking zone", "chicane", "hairpin", "sweeper", "esses",
    # Pit / paddock
    "pit lane", "pit box", "pit exit", "pit entry", "pit wall",
    "paddock", "garage", "tech inspection", "scrutineering", "weighbridge",
    # Ceremonial / media
    "starting grid", "grid walk", "victory lane", "podium", "trophy presentation",
    "press room", "media center",
]

MOTORSPORTS = Domain(
    name="motorsports",
    subject_types=_MOTORSPORTS_SUBJECT_TYPES,
    identifier_label="car number",
    settings=_MOTORSPORTS_SETTINGS,
    subject_field="is_motorsports",
    items_field="cars",
    item_fields=[
        ("type", "mid"),        # car category or subject: GT3 car, Porsche 911 GT3 Cup, driver portrait, etc.
        ("make", "mid"),        # Porsche, Ferrari, BMW M4, Chevrolet Camaro, etc.
        ("car_number", "high"), # race number read from markings on car
        ("series", "mid"),      # IMSA, Porsche Cup, F1, NASCAR Cup Series, IndyCar, etc.
        ("class", "mid"),       # GT3, GTD, PCR, Late Model, Cup, etc.
        ("livery", "phrase"),   # primary colors and major sponsor names
        ("details", "phrase"),  # specific features: mirror color, wing livery, damage, etc.
    ],
    prompt_fragments={
        "preamble": "This is a motorsports photograph (or might be).",
        "subject_qualifier": (
            "true only if the photo actually shows motorsports subject matter — "
            "race cars, racing drivers, pit crew, race officials, or race facilities"
        ),
        "item_singular": "car or person",
        "id_instruction": (
            "Fill car_number ONLY from numbers you can actually read on the car "
            "(door, nose, roof, or rear). Fill make (e.g. Porsche, Ferrari, BMW, "
            "Chevrolet) and class (e.g. GT3, GTD, PCR, Late Model, Cup Series) from "
            "body style, livery, and any visible badging. Fill series (e.g. IMSA, "
            "Porsche Cup, NASCAR Cup Series, IndyCar) if clearly identifiable. "
            "Record livery primary colors and major sponsor names visible on the car. "
            "Use details for specific features: mirror color, wing color, aero details, "
            "visible damage, distinctive markings. "
            "For people entries (driver portrait, pit crew, official, spectator), "
            "leave car_number, class, and series blank; fill make with team name if "
            "identifiable from suit or uniform."
        ),
        "era_examples": "modern, 2010s, 2000s, 1990s, 1980s, vintage/historic",
        "view_instruction": (
            "tracking shot (panned, car in motion), static/parked, pit stop action, "
            "starting grid, driver portrait, pit lane, paddock/garage, podium/victory lane, "
            "detail closeup, overhead/aerial, three-quarter front, three-quarter rear, "
            "broadside, head-on"
        ),
        "type_note": (
            "Porsche Club Racing: distinguish 911 GT3 Cup (rear-engine, tall rear wing, "
            "roll cage visible) from Cayman GT4 Clubsport (mid-engine, lower rear wing). "
            "NASCAR: distinguish Cup Series (Next Gen body, lower roofline) from Xfinity "
            "(taller greenhouse), Truck Series (pickup body), Super Late Model (full-bodied, "
            "no fenders), and Sportsman/Street Stock (production-based body). "
            "Prototypes: GTP/Hypercar (closed cockpit, aggressive aero) vs LMP2/LMP3 "
            "(open or semi-open cockpit, lower downforce). "
            "For people-only shots, create one item per person visible."
        ),
        "fallback_preamble": (
            "This is a motorsports photograph. Describe it for a searchable photo index "
            "using exact motorsports terminology — car type, manufacturer, race number if "
            "legible, racing series and class, livery colors and sponsor names, and setting. "
            "For people subjects, describe their role (driver, pit crew, official), any "
            "visible team affiliation, and what they are doing. "
            "If this is not a motorsports photo, describe what it actually shows with "
            "equal specificity. Write in plain sentences, no preamble."
        ),
    },
)


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

DOMAINS: dict[str, Domain] = {
    "railroad": RAILROAD,
    "naval": NAVAL,
    "armor": ARMOR,
    "aviation": AVIATION,
    "birds": BIRDS,
    "motorsports": MOTORSPORTS,
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
