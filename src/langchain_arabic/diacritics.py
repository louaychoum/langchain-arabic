"""Deterministic diacritics (tashkeel) restoration for Arabic LLM output.

Parses plain → diacritized word mappings from various sources (markdown, dict,
file) and applies longest-first string replacement on Arabic text.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Union


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


_FORMAT_PATTERNS = [
    # Most specific first — Unicode arrow (current format)
    re.compile(r"^-?[^\S\r\n]*(.+?)[^\S\r\n]*→[^\S\r\n]*(.+?)[^\S\r\n]*$", re.MULTILINE),
    # ASCII arrow
    re.compile(r"^-?[^\S\r\n]*(.+?)[^\S\r\n]*->[^\S\r\n]*(.+?)[^\S\r\n]*$", re.MULTILINE),
    # Tab-separated (unambiguous — tabs rarely appear elsewhere)
    re.compile(r"^-?[^\S\r\n]*(.+?)\t(.+?)[^\S\r\n]*$", re.MULTILINE),
    # Colon (most ambiguous — tried last, both sides must be Arabic)
    re.compile(r"^-?[^\S\r\n]*(.+?)[^\S\r\n]*:[^\S\r\n]*(.+?)[^\S\r\n]*$", re.MULTILINE),
]

_MARKS = r"[\u064b-\u0652\u0670]"


def _validate_mapping(mapping: Dict[str, str]) -> None:
    if any(
        not isinstance(key, str)
        or not re.sub(_MARKS, "", key).strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in mapping.items()
    ):
        raise ValueError("Diacritics mappings require nonempty string keys and values")


def _try_parse_with_patterns(text: str) -> Dict[str, str]:
    """Auto-detect delimiter format and extract diacritics mappings.

    Tries each format pattern in order of specificity (most specific first).
    Returns results from the first format that yields at least one valid
    mapping where both sides contain Arabic and plain != diacritized.
    """
    for pattern in _FORMAT_PATTERNS:
        mappings: Dict[str, str] = {}
        for match in pattern.finditer(text):
            plain = match.group(1).strip()
            diacritized = match.group(2).strip()
            if (
                plain
                and diacritized
                and plain != diacritized
                and _has_arabic(plain)
                and _has_arabic(diacritized)
            ):
                mappings[plain] = diacritized
        if mappings:
            return mappings
    return {}


def parse_diacritics_map(source: Union[str, Dict[str, str]]) -> Dict[str, str]:
    """Build a plain → diacritized mapping from *source*.

    Parameters
    ----------
    source
        - **dict**: validated and copied.
        - **str that is an existing file path**: read the file, then parse.
        - **str (markdown/text)**: auto-detects format and parses pairs.
          Supported formats: ``→``, ``->``, ``:``, tab-separated.

    Returns
    -------
    dict
        Mapping of plain Arabic words to their diacritized forms.
    """
    if isinstance(source, dict):
        _validate_mapping(source)
        return dict(source)

    text = source
    if os.path.isfile(source):
        with open(source, encoding="utf-8") as fh:
            text = fh.read()

    diacritics_map = _try_parse_with_patterns(text)

    if not diacritics_map and _has_arabic(text):
        import warnings

        warnings.warn(
            f"No diacritics mappings found in input ({len(text)} chars, "
            f"contains Arabic text). Supported formats: "
            f"'PLAIN → DIACRITIZED', 'PLAIN -> DIACRITIZED', "
            f"'PLAIN: DIACRITIZED', or tab-separated.",
            UserWarning,
            stacklevel=2,
        )

    return diacritics_map


def apply_diacritics(text: str, diacritics_map: Dict[str, str]) -> str:
    """Replace plain Arabic words with diacritized forms.

    Matches whole words/phrases, ignoring existing Arabic vowel marks.
    Longer keys take priority; replacements are applied in a single pass.

    Parameters
    ----------
    text
        The Arabic text to process.
    diacritics_map
        Mapping from :func:`parse_diacritics_map`.

    Returns
    -------
    str
        Text with diacritics applied.
    """
    if not diacritics_map or not text:
        return text

    _validate_mapping(diacritics_map)
    normalized = {re.sub(_MARKS, "", key): value for key, value in diacritics_map.items()}
    alternatives = [
        "".join(re.escape(char) + _MARKS + "*" for char in key)
        for key in sorted(normalized, key=len, reverse=True)
    ]
    pattern = r"(?<![\w\u064b-\u0652\u0670])(?:" + "|".join(alternatives) + r")(?!\w)"
    return re.sub(pattern, lambda match: normalized[re.sub(_MARKS, "", match[0])], text)


class DiacriticsProcessor:
    """Stateful wrapper that holds a diacritics map and exposes ``.process()``.

    Parameters
    ----------
    source
        Anything accepted by :func:`parse_diacritics_map`.

    Example
    -------
    >>> proc = DiacriticsProcessor({"تقنية": "تِقْنِيَة"})
    >>> proc.process("شركة تقنية")
    'شركة تِقْنِيَة'
    """

    def __init__(self, source: Union[str, Dict[str, str]]) -> None:
        self._map = parse_diacritics_map(source)

    @property
    def mapping(self) -> Dict[str, str]:
        """The current plain → diacritized mapping (read-only copy)."""
        return dict(self._map)

    def process(self, text: str) -> str:
        """Apply diacritics to *text*."""
        return apply_diacritics(text, self._map)
