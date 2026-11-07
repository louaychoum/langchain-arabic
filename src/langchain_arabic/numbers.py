"""Arabic and English number-to-word conversion with explicit numeric contexts."""

from __future__ import annotations

import re
import warnings
from typing import Set

from num2words import num2words

from langchain_arabic.dialects import SUPPORTED_DIALECTS, apply_dialect_number_overrides

_CONTEXTS = {"percentage", "currency_ar", "currency_en", "phone", "plain"}
_CURRENCIES = r"SAR|USD|AED|QAR"
_NUMBER = re.compile(
    rf"(?<![\w.٫])(?P<prefix>(?:{_CURRENCIES})[ \t]*)?"
    r"(?P<number>[+−-]?(?:\d{1,3}(?:[,٬]\d{3})+|\d+)(?:[.٫]\d+)?)"
    rf"(?P<suffix>[ \t]*(?:[%٪]|ريال|درهم|دينار|{_CURRENCIES}))?"
    r"(?!\w|[.٫]\d)"
)
_PHONE_CUE = re.compile(
    r"(?<!\w)(?:اتصل(?: على)?|هاتف|جوال|phone|tel|call)[ \t]*[:：]?[ \t]*$", re.IGNORECASE
)


def _validate_number_config(language: str, contexts: Set[str] | None, dialect: str) -> None:
    if language not in {"ar", "en"}:
        raise ValueError("language must be 'ar' or 'en'")
    if contexts is not None and not contexts <= _CONTEXTS:
        raise ValueError(f"Unknown number contexts: {sorted(contexts - _CONTEXTS)}")
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(f"Unsupported dialect {dialect!r}")


def _convert_single(number: str, context: str, language: str) -> str:
    cleaned = number.replace(",", "").replace("٬", "").replace("٫", ".")
    if context == "phone":
        return " ".join(num2words(int(d), lang=language) for d in cleaned if d.isdecimal())

    sign = ""
    if cleaned[0] in "+-−":
        if cleaned[0] in "-−":
            sign = "سالب " if language == "ar" else "minus "
        else:
            sign = "موجب " if language == "ar" else "plus "
        cleaned = cleaned[1:]
    integer, point, fraction = cleaned.partition(".")
    words = num2words(int(integer), lang=language)
    if point:
        separator = " فاصلة " if language == "ar" else " point "
        words += separator + " ".join(num2words(int(d), lang=language) for d in fraction)
    return sign + re.sub(r" {2,}", " ", words)


def convert_numbers_in_text(
    text: str,
    language: str = "ar",
    *,
    contexts: Set[str] | None = None,
    dialect: str = "msa",
) -> str:
    """Convert signed numbers, decimal digits, percentages, currencies and phones.

    Supports Western/Eastern Arabic digits, ``.``/``٫`` decimal separators,
    and grouped thousands (``1,000`` or ``١٬٠٠٠``). Fractional digits are read
    individually so precision and trailing zeros survive. Currency labels stay
    unchanged; grammatical currency agreement is not inferred.

    ``contexts`` selects percentage, currency_ar, currency_en, phone and/or plain.
    With all contexts enabled, phones require 7–15 ungrouped digits and a leading
    ``+`` or a preceding phone cue (e.g. ``اتصل على`` or ``phone:``).
    ``contexts={"phone"}`` explicitly reads bare 7–15 digit numbers as phones.
    Unsupported magnitudes are retained with a warning.
    """
    _validate_number_config(language, contexts, dialect)
    enabled = _CONTEXTS if contexts is None else contexts

    def replace(match: re.Match[str]) -> str:
        number = match["number"]
        prefix = match["prefix"] or ""
        suffix = match["suffix"] or ""
        label = suffix.strip()
        if label in {"%", "٪"}:
            context = "percentage"
        elif label in {"ريال", "درهم", "دينار"}:
            context = "currency_ar"
        elif prefix or label:
            context = "currency_en"
        elif re.fullmatch(r"\+?\d{7,15}", number) and (
            number.startswith("+")
            or _PHONE_CUE.search(re.sub(r"[\u064b-\u0652\u0670]", "", text[: match.start()]))
            or enabled == {"phone"}
        ):
            context = "phone"
        else:
            context = "plain"
        if context not in enabled:
            if "plain" not in enabled:
                return match[0]
            context = "plain"
        try:
            words = _convert_single(number, context, language)
        except (ValueError, OverflowError):
            warnings.warn("Number is outside the supported range; retained unchanged", stacklevel=2)
            return match[0]
        if context == "percentage":
            words += " بالمائة" if language == "ar" else " percent"
            suffix = ""
        if language == "ar":
            words = apply_dialect_number_overrides(words, dialect)
        return prefix + words + suffix

    return _NUMBER.sub(replace, text)


class NumbersProcessor:
    """Stateful number-to-word converter.

    Parameters
    ----------
    language
        ``"ar"`` or ``"en"``.
    contexts
        Which number contexts to handle. ``None`` = all.

    Example
    -------
    >>> proc = NumbersProcessor(language="ar")
    >>> proc.process("عام 2030")
    'عام ألفان و ثلاثون'
    """

    def __init__(
        self,
        language: str = "ar",
        contexts: Set[str] | None = None,
    ) -> None:
        _validate_number_config(language, contexts, "msa")
        self._language = language
        self._contexts = None if contexts is None else set(contexts)

    @property
    def language(self) -> str:
        return self._language

    def process(self, text: str) -> str:
        """Convert numbers in *text* to words."""
        return convert_numbers_in_text(text, self._language, contexts=self._contexts)
