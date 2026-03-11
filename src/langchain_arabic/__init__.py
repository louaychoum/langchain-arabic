"""langchain-arabic — Arabic text post-processing for LLM outputs."""

from langchain_arabic._version import __version__
from langchain_arabic.catt_backend import CATTBackend
from langchain_arabic.diacritics import (
    DiacriticsProcessor,
    apply_diacritics,
    parse_diacritics_map,
)
from langchain_arabic.dialects import (
    SUPPORTED_DIALECTS,
    apply_dialect_number_overrides,
    resolve_dialect_diacritics_map,
)
from langchain_arabic.numbers import NumbersProcessor, convert_numbers_in_text
from langchain_arabic.output_parser import ArabicTextOutputParser

__all__ = [
    "__version__",
    "apply_dialect_number_overrides",
    "apply_diacritics",
    "ArabicTextOutputParser",
    "CATTBackend",
    "convert_numbers_in_text",
    "DiacriticsProcessor",
    "NumbersProcessor",
    "parse_diacritics_map",
    "resolve_dialect_diacritics_map",
    "SUPPORTED_DIALECTS",
]
