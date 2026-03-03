"""Deterministic diacritics (tashkeel) restoration for Arabic LLM output.

Parses plain → diacritized word mappings from various sources (markdown, dict,
file) and applies longest-first string replacement on Arabic text.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Union

_ARROW_PATTERN = re.compile(r"^-\s+(.+?)\s*→\s*(.+?)\s*$", re.MULTILINE)


def parse_diacritics_map(source: Union[str, Dict[str, str]]) -> Dict[str, str]:
    """Build a plain → diacritized mapping from *source*.

    Parameters
    ----------
    source
        - **dict**: returned as-is (identity shortcut).
        - **str that is an existing file path**: read the file, then parse.
        - **str (markdown/text)**: parse ``- PLAIN → DIACRITIZED`` lines.

    Returns
    -------
    dict
        Mapping of plain Arabic words to their diacritized forms.
    """
    if isinstance(source, dict):
        return dict(source)

    text = source
    if os.path.isfile(source):
        with open(source, encoding="utf-8") as fh:
            text = fh.read()

    diacritics_map: Dict[str, str] = {}
    for match in _ARROW_PATTERN.finditer(text):
        plain = match.group(1).strip()
        diacritized = match.group(2).strip()
        if plain and diacritized and plain != diacritized:
            diacritics_map[plain] = diacritized

    return diacritics_map


def apply_diacritics(text: str, diacritics_map: Dict[str, str]) -> str:
    """Replace plain Arabic words with diacritized forms.

    Processes longer keys first to prevent partial-match conflicts
    (e.g. ``"علم الحاسوب"`` is replaced before ``"علم"``).

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

    for plain, diacritized in sorted(
        diacritics_map.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = text.replace(plain, diacritized)

    return text


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
