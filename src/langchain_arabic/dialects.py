"""Dialect-aware post-processing for Arabic text.

Provides dialect-specific overrides for number phrasing and
dialect-keyed diacritics map resolution.  Currently supports
``"msa"`` (Modern Standard Arabic) and ``"gulf"`` (Khaleeji).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Union, cast

from langchain_arabic.diacritics import parse_diacritics_map

SUPPORTED_DIALECTS: frozenset[str] = frozenset({"msa", "gulf"})

# Post-processing substitutions applied to num2words MSA output.
# Each dialect maps MSA words to their dialect equivalents.
_DIALECT_NUMBER_OVERRIDES: Dict[str, Dict[str, str]] = {
    "msa": {},
    "gulf": {
        "مئتان": "ميتين",
        "مائتان": "ميتين",
        "بالمائة": "بالمية",
        "مائة": "مية",
    },
}


def _validate_dialect(dialect: str) -> None:
    """Raise ``ValueError`` if *dialect* is not in :data:`SUPPORTED_DIALECTS`."""
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(
            f"Unsupported dialect {dialect!r}. Choose from: {sorted(SUPPORTED_DIALECTS)}"
        )


def apply_dialect_number_overrides(text: str, dialect: str) -> str:
    """Apply dialect-specific word substitutions to number text.

    Parameters
    ----------
    text
        Text containing Arabic number words (output of ``num2words``).
    dialect
        Target dialect (``"msa"`` or ``"gulf"``).

    Returns
    -------
    str
        Text with dialect-appropriate number phrasing.
    """
    _validate_dialect(dialect)
    overrides = _DIALECT_NUMBER_OVERRIDES.get(dialect, {})
    if not overrides or not text:
        return text
    for msa_word, dialect_word in sorted(
        overrides.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        text = text.replace(msa_word, dialect_word)
    return text


def resolve_dialect_diacritics_map(
    diacritics_map: Mapping[str, Union[str, Dict[str, str]]],
    dialect: str,
) -> Dict[str, str]:
    """Resolve a flat or dialect-keyed diacritics map for *dialect*.

    A **flat map** (all values are ``str``) is returned as-is regardless
    of dialect — it is dialect-agnostic.

    A **dialect-keyed map** (values are ``dict``) selects the sub-dict
    matching *dialect*, falling back to ``"msa"``, then to an empty dict.

    Parameters
    ----------
    diacritics_map
        Either ``{"word": "diacritized"}`` (flat) or
        ``{"msa": {...}, "gulf": {...}}`` (dialect-keyed).
    dialect
        Target dialect.

    Returns
    -------
    dict
        A flat ``{plain: diacritized}`` mapping.
    """
    _validate_dialect(dialect)
    if not diacritics_map:
        return {}

    all_dicts = all(isinstance(v, dict) for v in diacritics_map.values())
    all_strs = all(isinstance(v, str) for v in diacritics_map.values())
    if not (all_dicts or all_strs):
        raise ValueError(
            "diacritics_map values must be all str (flat) or all dict (dialect-keyed), "
            "got mixed types."
        )

    if all_dicts:
        # Dialect-keyed map
        keyed = cast(Mapping[str, Dict[str, str]], diacritics_map)
        return parse_diacritics_map(keyed.get(dialect, keyed.get("msa", {})))

    # Flat map — return as-is
    return parse_diacritics_map(dict(cast(Mapping[str, str], diacritics_map)))
