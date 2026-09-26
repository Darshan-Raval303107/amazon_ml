"""
Text normalization module for Business Entity Resolution.

Provides four public functions used by blocking, feature engineering,
MiniLM embeddings, and CatBoost:

    normalize_business_name(text)    -> str
    normalize_business_address(text) -> str
    normalize_country(text)          -> str
    normalize_record(record)         -> dict

Design principles
-----------------
* Conservative: we normalise *surface noise* (casing, punctuation, redundant
  whitespace, well-known abbreviations) but deliberately avoid aggressive token
  removal or stemming, which would destroy information needed to distinguish
  near-duplicate entities (e.g. "ABC Tech" vs "ABC Technologies").
* Pure Python / stdlib + unicodedata: no downloads, no external APIs, no ML
  models are loaded here.
* Deterministic & idempotent: running the same function twice on its own output
  returns the same result.
* Reusable: all four functions are stateless module-level callables.
"""

import re
import unicodedata
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Legal / corporate entity-type tokens that appear in many different notations.
# Mapping: normalised abbreviation token -> canonical expanded form.
# Keys are *already lowercased* and punctuation-stripped.
# These are applied after the string is lowercased and punctuation is removed,
# using whole-word boundary replacement, so "pvt" only matches the token "pvt"
# and not substrings inside longer words.
_LEGAL_SUFFIX_MAP: dict = {
    # Private / Public
    "pvt":          "private",
    "priv":         "private",
    "pub":          "public",
    # Limited / LLC / LLP
    "ltd":          "limited",
    "llc":          "llc",
    "llp":          "llp",
    "lp":           "lp",
    "plc":          "plc",
    # Corporation / Incorporated / Company
    "corp":         "corporation",
    "inc":          "incorporated",
    "co ltd":       "company limited",
    "co":           "company",
    # Other international suffixes
    "srl":          "srl",
    "gmbh":         "gmbh",
    "ag":           "ag",
    "bv":           "bv",
    "nv":           "nv",
    "sa":           "sa",
    "pty":          "proprietary",
    "sdn bhd":      "sendirian berhad",
    "bhd":          "berhad",
    "pte":          "private",
    "opc":          "one person company",
}

# Address abbreviations: street type tokens -> expanded form.
_ADDRESS_ABBR_MAP: dict = {
    # Street types
    "st":    "street",
    "rd":    "road",
    "ave":   "avenue",
    "blvd":  "boulevard",
    "dr":    "drive",
    "ln":    "lane",
    "ct":    "court",
    "crt":   "court",
    "pl":    "place",
    "sq":    "square",
    "hwy":   "highway",
    "fwy":   "freeway",
    "expy":  "expressway",
    "pkwy":  "parkway",
    "trl":   "trail",
    "aly":   "alley",
    "xing":  "crossing",
    # Unit / Floor designators
    "fl":    "floor",
    "flr":   "floor",
    "ste":   "suite",
    "apt":   "apartment",
    "bldg":  "building",
    "rm":    "room",
    # Directions
    "ne":    "northeast",
    "nw":    "northwest",
    "se":    "southeast",
    "sw":    "southwest",
    "n":     "north",
    "s":     "south",
    "e":     "east",
    "w":     "west",
    # India-specific
    "mg":    "mahatma gandhi",
    "opp":   "opposite",
    "nr":    "near",
    "soc":   "society",
    # PO Box normalisation
    "po box":  "po box",
    "p o box": "po box",
}

# Characters treated as punctuation (to be replaced with space).
# Hyphens (-) are kept because they carry meaning in "Hewlett-Packard" etc.
# Ampersands (&) are handled separately before this step.
_PUNCT_TO_SPACE_RE = re.compile(r"[^\w\s\-]")

# Collapse runs of whitespace to a single space.
_WHITESPACE_RE = re.compile(r"\s+")

# Sentinel strings considered equivalent to "missing".
_MISSING_SENTINELS = frozenset({"nan", "none", "null", "n/a", "na", "-", "--", ""})


def _make_word_re(token: str) -> re.Pattern:
    """Return a compiled whole-word regex for *token*."""
    return re.compile(r"(?<!\w)" + re.escape(token) + r"(?!\w)")


# Pre-compile replacement patterns, sorted longest-key-first so that
# multi-word keys like "co ltd" are matched before "co" or "ltd".
_LEGAL_PATTERNS = [
    (_make_word_re(k), v)
    for k, v in sorted(_LEGAL_SUFFIX_MAP.items(), key=lambda x: -len(x[0]))
]

_ADDRESS_PATTERNS = [
    (_make_word_re(k), v)
    for k, v in sorted(_ADDRESS_ABBR_MAP.items(), key=lambda x: -len(x[0]))
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_str(text: object) -> Optional[str]:
    """
    Coerce *text* to a Python str, returning None for genuinely missing values.

    Treats None, float NaN (from pandas), and common sentinel strings
    ("nan", "none", "null", "n/a", "na", "-", "--", "") as missing.
    """
    if text is None:
        return None
    if isinstance(text, float) and math.isnan(text):
        return None
    s = str(text).strip()
    if s.lower() in _MISSING_SENTINELS:
        return None
    return s


def _unicode_normalise(text: str) -> str:
    """
    Apply Unicode NFKC normalisation.

    NFKC decomposes compatibility characters (ligatures like 'fi', full-width
    digits, superscripts) and recomposes canonical equivalents, ensuring
    consistent character representation across differently-encoded sources.
    """
    return unicodedata.normalize("NFKC", text)


def _strip_punctuation_to_space(text: str) -> str:
    """
    Replace punctuation with spaces, keeping hyphens and word/digit chars.

    Hyphens are preserved because they are semantically meaningful in many
    business names (e.g. "Coca-Cola") and address ranges ("Suite 100-200").
    """
    return _PUNCT_TO_SPACE_RE.sub(" ", text)


def _normalise_whitespace(text: str) -> str:
    """Collapse any run of whitespace to one space and strip ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _apply_token_map(text: str, patterns: list) -> str:
    """
    Apply (compiled_pattern, replacement) pairs to *text* in order.

    Patterns must be sorted longest-key-first to prevent partial replacements.
    """
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_business_name(text: object) -> str:
    """
    Normalize a raw business name string for entity resolution.

    Steps applied in order
    ----------------------
    1. **Missing value handling** - Returns ``""`` for None, NaN, blank, or
       sentinel values ("N/A", "none", "-", etc.).
    2. **Lowercase** - Eliminates surface variation due to capitalisation.
    3. **Unicode NFKC normalisation** - Resolves encoding inconsistencies
       (ligatures, full-width chars) without discarding non-ASCII characters.
    4. **Ampersand normalisation** - Replaces ``&`` with ``and`` so that
       "Smith & Jones" and "Smith and Jones" compare equally.
    5. **Punctuation stripping** - Removes periods, commas, slashes,
       parentheses, etc. by replacing them with spaces; hyphens are kept.
    6. **Whitespace normalisation** - Collapses consecutive whitespace to a
       single space and strips leading/trailing whitespace.
    7. **Legal suffix expansion** - Expands abbreviated entity-type tokens
       (e.g. "pvt" -> "private", "ltd" -> "limited", "corp" -> "corporation")
       using whole-word matching so only exact tokens are expanded.
    8. **Final whitespace cleanup** - Second pass after token expansion.

    Parameters
    ----------
    text : object
        Raw business name (str, None, float NaN, or any other type).

    Returns
    -------
    str
        Normalised business name, or ``""`` if the input was missing.

    Examples
    --------
    >>> normalize_business_name("ABC Technologies Pvt Ltd")
    'abc technologies private limited'
    >>> normalize_business_name("ABC TECHNOLOGIES PVT. LTD.")
    'abc technologies private limited'
    >>> normalize_business_name("Smith & Jones Corp.")
    'smith and jones corporation'
    >>> normalize_business_name(None)
    ''
    """
    raw = _to_str(text)
    if raw is None:
        return ""

    s = raw.lower()                              # Step 2
    s = _unicode_normalise(s)                    # Step 3
    s = re.sub(r"\s*&\s*", " and ", s)           # Step 4
    s = _strip_punctuation_to_space(s)           # Step 5
    s = _normalise_whitespace(s)                 # Step 6
    s = _apply_token_map(s, _LEGAL_PATTERNS)     # Step 7
    s = _normalise_whitespace(s)                 # Step 8

    return s


def normalize_business_address(text: object) -> str:
    """
    Normalize a raw business address string for entity resolution.

    Steps applied in order
    ----------------------
    1. **Missing value handling** - Returns ``""`` for missing/sentinel input.
    2. **Lowercase** - Reduces variation in address tokens.
    3. **Unicode NFKC normalisation** - Resolves encoding inconsistencies.
    4. **Ampersand normalisation** - ``&`` -> ``and``.
    5. **Punctuation stripping** - Periods, commas, slashes, etc. to spaces;
       hyphens preserved.
    6. **Whitespace normalisation**.
    7. **Address abbreviation expansion** - Expands known street-type tokens
       (``st``->``street``, ``rd``->``road``, ``ave``->``avenue``), directional
       tokens (``n``->``north``, ``sw``->``southwest``), and unit designators
       (``fl``->``floor``, ``apt``->``apartment``) using whole-word matching.
    8. **Final whitespace cleanup**.

    No geocoding, no external APIs, no postal-code stripping.
    Postal/PIN codes are deliberately preserved as strong blocking signals.

    Parameters
    ----------
    text : object
        Raw address string.

    Returns
    -------
    str
        Normalised address string, or ``""`` if missing.

    Examples
    --------
    >>> normalize_business_address("12 MG Road, Bengaluru")
    '12 mahatma gandhi road bengaluru'
    >>> normalize_business_address("105 ELM ST, MORGANTON, NC")
    '105 elm street morganton nc'
    >>> normalize_business_address("914 PIERPONT AVE, CLEVELAND, OH")
    '914 pierpont avenue cleveland oh'
    """
    raw = _to_str(text)
    if raw is None:
        return ""

    s = raw.lower()                                # Step 2
    s = _unicode_normalise(s)                      # Step 3
    s = re.sub(r"\s*&\s*", " and ", s)             # Step 4
    s = _strip_punctuation_to_space(s)             # Step 5
    s = _normalise_whitespace(s)                   # Step 6
    s = _apply_token_map(s, _ADDRESS_PATTERNS)     # Step 7
    s = _normalise_whitespace(s)                   # Step 8

    return s


def normalize_country(text: object) -> str:
    """
    Normalize a raw country string.

    Rules
    -----
    * Returns ``""`` for missing / sentinel values.
    * Applies Unicode NFKC and lowercase.
    * Strips leading/trailing whitespace; collapses internal whitespace.
    * Does **not** expand abbreviations (e.g. "US" stays "us") because the
      mapping is open-set and aggressive expansion risks errors for unseen codes.
    * Does **not** hard-code or filter to a fixed country list.

    Parameters
    ----------
    text : object
        Raw country string.

    Returns
    -------
    str
        Normalised country string (lowercase, NFKC, whitespace collapsed),
        or ``""`` if missing.

    Examples
    --------
    >>> normalize_country("United States")
    'united states'
    >>> normalize_country("INDIA")
    'india'
    >>> normalize_country("  France  ")
    'france'
    >>> normalize_country(None)
    ''
    """
    raw = _to_str(text)
    if raw is None:
        return ""

    s = raw.lower()
    s = _unicode_normalise(s)
    s = _normalise_whitespace(s)
    return s


def normalize_record(record: dict) -> dict:
    """
    Normalize all textual fields of a single entity record.

    Applies :func:`normalize_business_name`, :func:`normalize_business_address`,
    and :func:`normalize_country` to the corresponding fields and returns a
    **new dictionary** without modifying the original *record*.

    Any additional keys (e.g. ``entity_id``) are copied unchanged.

    Parameters
    ----------
    record : dict
        A record dict, typically with keys:
        ``entity_id``, ``business_name``, ``business_address``, ``country``.

    Returns
    -------
    dict
        New dict with normalised values for the three text fields.

    Examples
    --------
    >>> r = {"entity_id": "S1-001", "business_name": "ABC Corp.",
    ...      "business_address": "12 MG Rd", "country": "India"}
    >>> norm = normalize_record(r)
    >>> norm["business_name"]
    'abc corporation'
    >>> norm["country"]
    'india'
    >>> r["business_name"]   # original unchanged
    'ABC Corp.'
    """
    normalised = dict(record)  # shallow copy so original is untouched

    normalised["business_name"] = normalize_business_name(
        record.get("business_name")
    )
    normalised["business_address"] = normalize_business_address(
        record.get("business_address")
    )
    normalised["country"] = normalize_country(
        record.get("country")
    )

    return normalised


# ---------------------------------------------------------------------------
# Self-contained demonstration
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Smoke-test and demonstrate normalization on representative sample strings.

    Run with:  python -m business_entity_resolution.src.normalization
    """
    SEP = "=" * 68

    # ---- Business Names ----
    print(SEP)
    print("Business Name Normalization")
    print(SEP)

    name_examples = [
        # Task-specified examples
        "ABC Technologies Pvt Ltd",
        "ABC TECHNOLOGIES PVT. LTD.",
        "ABC Tech",
        # Real dataset samples
        "B+ Retail Inc",
        "Delta Tetlecommunication Inc",
        "-- Holloway Peak Inc Seafood",
        "Summit Inc",
        "International South Consultants Private Ltd",
        "Pvt. EFS Print Ventures Ltd.",
        "Smith & Jones Corp.",
        "Walmart Inc.",
        "WALMART, INC",
        None,
        "N/A",
        "",
    ]

    for name in name_examples:
        out = normalize_business_name(name)
        print(f"  IN : {name!r}")
        print(f"  OUT: {out!r}")
        print()

    # ---- Addresses ----
    print(SEP)
    print("Business Address Normalization")
    print(SEP)

    addr_examples = [
        # Task-specified examples
        "12 MG Road, Bengaluru",
        "12 M.G. Road Bengaluru",
        "Near City Mall, MG Road",
        # Real dataset samples
        "105 ELM ST, MORGANTON, NC",
        "914 PIERPONT AVE, CLEVELAND, OH",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
        "1795 Westchester Dr, High Point, NC",
        "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban",
        "1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio",
        "2100 Cameron Dr, Unit APARTMENT G, Dundalk, MD",
        None,
        "",
    ]

    for addr in addr_examples:
        out = normalize_business_address(addr)
        print(f"  IN : {addr!r}")
        print(f"  OUT: {out!r}")
        print()

    # ---- Countries ----
    print(SEP)
    print("Country Normalization")
    print(SEP)

    country_examples = [
        "United States", "united states", "US", "u.s.", "U.S.A.",
        "India", "INDIA", "india",
        "France", "FRANCE",
        "Germany", "Netherlands", "Malaysia", "Singapore",
        None, "N/A", "",
    ]

    for country in country_examples:
        out = normalize_country(country)
        print(f"  IN : {str(country):22s}  OUT: {out!r}")

    # ---- Records ----
    print()
    print(SEP)
    print("Record Normalization")
    print(SEP)

    records = [
        {
            "entity_id": "S1-925783039",
            "business_name": "Orelee's Barbershop",
            "business_address": "1795 Westchester Drive, High Point, NC",
            "country": "US",
        },
        {
            "entity_id": "S3-859268022",
            "business_name": "International South Consultants Private Ltd",
            "business_address": None,
            "country": "India",
        },
        {
            "entity_id": "S2-163963287",
            "business_name": "Summit Inc",
            "business_address": "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
            "country": "US",
        },
    ]

    for rec in records:
        norm = normalize_record(rec)
        print(f"  entity_id       : {norm['entity_id']}")
        print(f"  business_name   : {norm['business_name']!r}")
        print(f"  business_address: {norm['business_address']!r}")
        print(f"  country         : {norm['country']!r}")
        print()


if __name__ == "__main__":
    _run_demo()
