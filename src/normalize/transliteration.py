"""Script detection and deterministic transliteration module for cross-script entity resolution.

Supports Unicode-based script detection and romanization of Indic scripts
(Devanagari, Gujarati, Tamil, Bengali, Kannada, Telugu, Malayalam) to Latin.
"""

import re
import unicodedata
from typing import Optional

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_LIB_AVAILABLE = True
except ImportError:
    INDIC_LIB_AVAILABLE = False


# Unicode block definitions (ranges)
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

# Mapping to indic_transliteration sanscript scheme constants
SANSCRIPT_MAP = {}
if INDIC_LIB_AVAILABLE:
    SANSCRIPT_MAP = {
        "devanagari": sanscript.DEVANAGARI,
        "bengali": sanscript.BENGALI,
        "gurmukhi": sanscript.GURMUKHI,
        "gujarati": sanscript.GUJARATI,
        "oriya": sanscript.ORIYA,
        "tamil": sanscript.TAMIL,
        "telugu": sanscript.TELUGU,
        "kannada": sanscript.KANNADA,
        "malayalam": sanscript.MALAYALAM,
    }


def _get_char_script(char: str) -> Optional[str]:
    """Identify script of a single character by Unicode code point."""
    cp = ord(char)
    # Check ASCII / Latin
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
        return "latin"
    for script_name, (start, end) in SCRIPT_RANGES.items():
        if start <= cp <= end:
            return script_name
    return None


def detect_script(text: Optional[str]) -> str:
    """Detect primary script of text. Fast-paths ASCII strings."""
    if text is None:
        return "empty"
    
    text_str = str(text).strip()
    if not text_str:
        return "empty"

    # Ultra-fast path for standard ASCII Latin
    if text_str.isascii():
        # Verify it has alphanumeric chars
        if any(c.isalnum() for c in text_str):
            return "latin"
        return "unknown"

    scripts_found = set()
    for char in text_str:
        s = _get_char_script(char)
        if s is not None:
            scripts_found.add(s)

    if not scripts_found:
        return "unknown"

    if len(scripts_found) == 1:
        return next(iter(scripts_found))

    return "mixed"


def get_script_flag(text: Optional[str]) -> str:
    """Convenience wrapper returning the detect_script() result for dataframe columns."""
    return detect_script(text)


def _transliterate_chunk(text: str, script: str) -> str:
    """Transliterate a single contiguous chunk of non-Latin script."""
    if not INDIC_LIB_AVAILABLE or script not in SANSCRIPT_MAP:
        return text
    try:
        src_scheme = SANSCRIPT_MAP[script]
        # Transliterate to ITRANS (standard ASCII romanization)
        romanized = transliterate(text, src_scheme, sanscript.ITRANS)
        return romanized
    except Exception:
        return text


def transliterate_to_latin(text: Optional[str], script: Optional[str] = None) -> str:
    """Transliterate non-Latin or mixed script text to Latin."""
    if text is None:
        return ""
    
    text_str = str(text)
    if not text_str.strip():
        return ""

    if text_str.isascii():
        return text_str

    if script is None:
        script = detect_script(text_str)

    if script in ("latin", "unknown", "empty"):
        return text_str

    if not INDIC_LIB_AVAILABLE:
        return text_str

    if script in SANSCRIPT_MAP:
        return _transliterate_chunk(text_str, script)

    if script == "mixed":
        chunks = []
        current_chunk = []
        current_script = None

        for char in text_str:
            c_script = _get_char_script(char)
            if c_script is None:
                current_chunk.append(char)
            elif c_script == current_script:
                current_chunk.append(char)
            else:
                if current_chunk:
                    chunks.append(("".join(current_chunk), current_script))
                current_chunk = [char]
                current_script = c_script

        if current_chunk:
            chunks.append(("".join(current_chunk), current_script))

        transliterated_parts = []
        for chunk_text, chunk_script in chunks:
            if chunk_script in SANSCRIPT_MAP:
                transliterated_parts.append(_transliterate_chunk(chunk_text, chunk_script))
            else:
                transliterated_parts.append(chunk_text)

        return "".join(transliterated_parts)

    return text_str
