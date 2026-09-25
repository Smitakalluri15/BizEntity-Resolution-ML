"""Country normalization module.

Applies open-set, country-agnostic normalization:
- Handles null/empty and placeholder values
- Lowercases
- Strips leading/trailing whitespace and non-alphanumeric noise

DO NOT add country allowlists or fixed mapping dictionaries here, as the test set
contains unseen countries (e.g. France) not present in the training set.
"""

import re
from typing import Optional


def normalize_country(country: Optional[str]) -> str:
    """Normalize country string in an open-set, country-agnostic manner."""
    if country is None:
        return ""
    text = str(country).strip()
    if not text or text.lower() in ('nan', 'null', '<null>', 'n/a', 'none'):
        return ""
    # Lowercase
    text = text.lower()
    # Strip non-alphanumeric except space and hyphen
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
