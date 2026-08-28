"""
Central configuration for the SIF Precursor Detection Engine.
"""

# ── Model Configuration ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # Sentence transformer for embeddings
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"   # Zero-shot classification backbone

# ── Classification Thresholds ─────────────────────────────────────────────────
SIF_SCORE_THRESHOLD    = 0.50   # Above this → SIF-potential
LSR_SIMILARITY_THRESHOLD = 0.30  # Minimum cosine similarity to tag an LSR rule
HIGH_CONFIDENCE_CUTOFF = 0.75   # SIF score above this → "HIGH" confidence
LOW_CONFIDENCE_CUTOFF  = 0.40   # SIF score below this → "LOW" confidence

# ── SIF Ensemble Weights ──────────────────────────────────────────────────────
WEIGHT_LLM     = 0.60   # Weight for zero-shot LLM probability
WEIGHT_KEYWORD = 0.40   # Weight for keyword/energy-hazard score

# ── IOGP Life-Saving Rules ────────────────────────────────────────────────────
LIFE_SAVING_RULES = {
    "Energy Isolation": {
        "description": (
            "Verify isolation and zero energy before work begins. "
            "Lockout tagout LOTO de-energize electrical isolation valve isolation."
        ),
        "keywords": [
            "loto", "lockout", "tagout", "lock out", "tag out", "isolation",
            "zero energy", "de-energize", "de energize", "isolate", "isolation certificate",
            "energy control", "electrical isolation", "valve isolation", "blinding",
            "blinds and spades", "line break", "residual energy",
        ],
        "color": "#e74c3c",
        "icon": "⚡",
    },
    "Confined Space": {
        "description": (
            "Obtain authorization before entering a confined space. "
            "Confined space tank vessel pit manhole atmospheric test H2S oxygen gas monitor standby man."
        ),
        "keywords": [
            "confined space", "confined space entry", "tank entry", "vessel entry",
            "pit", "manhole", "sump", "h2s", "hydrogen sulphide", "hydrogen sulfide",
            "atmospheric test", "gas monitor", "standby man", "attendant",
            "oxygen deficiency", "toxic atmosphere", "rescue plan", "air supply",
        ],
        "color": "#9b59b6",
        "icon": "🚪",
    },
    "Hot Work": {
        "description": (
            "Control flammables and ignition sources during hot work. "
            "Welding cutting grinding spark flammable gas test fire watch."
        ),
        "keywords": [
            "welding", "weld", "cutting", "grinding", "grind", "spark",
            "flame", "hot work", "hot work permit", "flammable", "combustible",
            "gas test", "fire watch", "firewatch", "fire extinguisher",
            "ignition source", "open flame", "torch", "burning",
        ],
        "color": "#e67e22",
        "icon": "🔥",
    },
    "Line of Fire": {
        "description": (
            "Keep yourself and others out of the line of fire. "
            "Dropped object struck by caught in pinch point projectile suspended load."
        ),
        "keywords": [
            "line of fire", "dropped object", "falling object", "struck by",
            "caught in", "caught between", "pinch point", "projectile",
            "suspended load", "overhead", "exclusion zone", "danger zone",
            "high pressure", "pressure release", "blowout", "ejection",
            "whipping hose", "snap back", "stored energy release",
        ],
        "color": "#c0392b",
        "icon": "🎯",
    },
    "Working at Height": {
        "description": (
            "Protect yourself against a fall when working at height. "
            "Scaffold ladder fall harness edge protection roof platform."
        ),
        "keywords": [
            "working at height", "height", "fall", "falling", "scaffold",
            "scaffolding", "ladder", "roof", "elevated", "harness",
            "fall protection", "edge protection", "guardrail", "safety net",
            "personal fall arrest", "lanyard", "anchor point", "platform",
        ],
        "color": "#27ae60",
        "icon": "🪜",
    },
    "Safe Mechanical Lifting": {
        "description": (
            "Plan lifting operations and control the area. "
            "Crane sling rigging load overhead lifting plan banksman."
        ),
        "keywords": [
            "crane", "lifting", "lift", "sling", "slings", "rigging",
            "suspended load", "overhead load", "lifting plan", "banksman",
            "rigger", "load chart", "rated capacity", "overload",
            "wire rope", "shackle", "hook", "load radius", "boom",
        ],
        "color": "#2980b9",
        "icon": "🏗️",
    },
    "Work Authorisation": {
        "description": (
            "Work with a valid permit when required. "
            "Permit to work PTW authorization JSA job safety analysis."
        ),
        "keywords": [
            "permit to work", "ptw", "work permit", "permit", "authorization",
            "authorisation", "jsa", "job safety analysis", "risk assessment",
            "toolbox talk", "no permit", "expired permit", "unsigned permit",
            "incomplete permit", "without authorization", "unauthorized",
        ],
        "color": "#16a085",
        "icon": "📋",
    },
    "Driving": {
        "description": (
            "Follow safe driving rules. "
            "Vehicle speeding seatbelt fatigue collision defensive driving."
        ),
        "keywords": [
            "driving", "vehicle", "car", "truck", "speeding", "speed",
            "seatbelt", "seat belt", "fatigue", "drowsy", "collision",
            "accident", "road", "defensive driving", "mobile phone",
            "distracted", "reversing", "blind spot", "journey management",
        ],
        "color": "#8e44ad",
        "icon": "🚗",
    },
    "Bypassing Safety Controls": {
        "description": (
            "Obtain authorization before overriding or disabling safety controls. "
            "Override bypass defeat disable interlock safety device."
        ),
        "keywords": [
            "bypass", "bypassing", "override", "overriding", "defeat",
            "disable", "disabling", "interlock", "safety device",
            "safety system", "inhibit", "inhibited", "defeated",
            "removed guard", "guard removal", "safety valve", "relief valve bypassed",
        ],
        "color": "#d35400",
        "icon": "🔓",
    },
}

# ── Oil-Field Abbreviation Expansions ─────────────────────────────────────────
OIL_FIELD_ABBREVIATIONS = {
    "loto":   "lockout tagout",
    "ptw":    "permit to work",
    "h2s":    "hydrogen sulphide",
    "co2":    "carbon dioxide",
    "jsa":    "job safety analysis",
    "ppe":    "personal protective equipment",
    "hsse":   "health safety security environment",
    "ua":     "unsafe act",
    "uc":     "unsafe condition",
    "moc":    "management of change",
    "toolbox": "toolbox talk",
    "tbm":    "toolbox meeting",
    "lti":    "lost time injury",
    "rwc":    "restricted work case",
    "mtc":    "medical treatment case",
    "fac":    "first aid case",
    "esd":    "emergency shutdown",
    "psv":    "pressure safety valve",
    "prv":    "pressure relief valve",
    "lopa":   "layer of protection analysis",
    "hazop":  "hazard and operability study",
    "swl":    "safe working load",
    "slr":    "sling load rating",
    "wll":    "working load limit",
    "bop":    "blowout preventer",
    "scba":   "self contained breathing apparatus",
    "msds":   "material safety data sheet",
    "sds":    "safety data sheet",
}

# ── High-Energy Hazard Keywords (for keyword scoring) ────────────────────────
HIGH_ENERGY_KEYWORDS = [
    "electrical", "electrocution", "electric shock", "high voltage", "live panel", "busbar", "415v", "energised", "energized",
    "pressure", "high pressure", "pressurised", "pressurized", "overpressure", "bar", "psi", "flange", "valve", "pipe", "pipeline", "gas",
    "explosion", "fire", "flash fire", "ignition", "ignited", "flammable", "combustible", "flare", "hydrocarbon",
    "chemical", "toxic", "poisonous", "h2s", "hydrogen sulphide", "asphyxiation", "oxygen deficiency", "dizzy", "unconscious",
    "fall", "falling", "height", "scaffold", "scaffolding", "ladder", "roof", "gravity", "fragile sheet",
    "crush", "crushing", "pinch", "caught", "impeller", "rotating", "coupling", "shaft",
    "dropped object", "struck", "struck by", "impact", "bop", "crane", "lifting", "load", "overload", "sling", "rigging", "winch", "wire rope",
    "radiation", "radioactive",
    "kinetic energy", "rotating equipment", "moving machinery",
    "speeding", "collision", "vehicle", "truck", "tanker", "rollover",
]

# ── Critical/Near-Fatal Signal Phrases ───────────────────────────────────────
NEAR_FATAL_SIGNALS = [
    "could have been fatal", "could have been killed", "nearly fatal",
    "critical incident", "high potential", "hipot", "high pot",
    "life threatening", "serious injury", "fatality potential",
    "near miss", "near-miss", "close call", "narrowly avoided",
    "almost", "just missed", "luckily", "fortunately",
    "injuring", "injured", "hospitalised", "hospitalized", "unconscious", "loss of consciousness",
    "electric shock", "burn", "burned", "burning", "degloving", "dropped onto", "fell onto",
    "flash fire", "pressure release", "blowout", "explosion",
]

# ── Barrier Failure Signals ───────────────────────────────────────────────────
BARRIER_FAILURE_SIGNALS = [
    "not wearing", "without ppe", "no harness", "without harness", "no permit",
    "expired permit", "incomplete permit", "invalid permit", "without permit", "without a permit", "without a valid permit",
    "bypass", "bypassed", "bypassing", "override", "overriding", "wired shut", "inhibited", "defeat", "defeated",
    "missing guard", "guard removed", "removed guard", "no gas test", "not tested", "without conducting", "without completing",
    "not isolated", "isolation not", "not completed", "loto was not", "no loto", "without loto", "lockout tagout", "not locked out",
    "unauthorized", "without authorization", "no training", "untrained", "inadequate", "defective", "damaged equipment",
    "no standby man", "rescue equipment was not", "no fire watcher", "no fire watch", "empty fire extinguisher",
    "no tool tethering", "no toe-boards", "no exclusion zone", "past its rated inspection date", "without fall protection",
    "no scaffold inspection", "without a buddy", "ladder was not secured", "no fall arrest", "no safety net", "no lift plan",
    "without a valid lifting plan", "banksman was not in position", "shackle pin was unscrewed", "no jsa", "without re-authorization",
    "not wearing a seatbelt", "no seatbelt", "mobile phone", "without a banksman", "reversing camera was defective",
    "journey management plan had not been followed",
]

# ── Database Configuration ────────────────────────────────────────────────────
DB_PATH = "sif_reports.duckdb"

# ── UI Configuration ──────────────────────────────────────────────────────────
PAGE_TITLE = "SIF Precursor Detection — Oil India Limited"
PAGE_ICON  = "🛢️"
SIDEBAR_LOGO = "OIL"
