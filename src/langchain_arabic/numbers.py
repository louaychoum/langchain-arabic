"""Arabic and English number-to-word conversion with explicit numeric contexts."""

from __future__ import annotations

import re
import threading
import warnings
from typing import Callable, Set

from num2words import num2words

from langchain_arabic.diacritics import _MARKS
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
# Attached conjunction (و، ف) or a tatweel-marked preposition (بـ، لـ، كـ, also وبـ)
# directly before a number: و10، بـ50، وبـ3. A bare ب/ك/ل before digits stays a code
# because letter+digit codes such as فيتامين ب12 or وب12 are common.
_CLITIC = re.compile(r"(?<![\w./:+-])(?:[وف]ـ?|[وف]?[بلك]ـ)(?=[+−-]?\d)")
# shortcut: cues more than ~60 characters before the number (e.g. long padding) are
# ignored; scanning only this tail keeps long texts linear.
_CUE_WINDOW = 64
# num2words 0.5.14 (Arabic) repeats the scale word for groups 101-901: 101000 is
# "مائة و ألف ألف". Rewrite as "مائة ألف و ألف" (construct dual مئتا before a scale word).
_REPEATED_SCALE = re.compile(r"(\S+) و (\S+) \2(?!\S)")
# num2words' Arabic converter keeps per-call state on a shared instance.
_NUM2WORDS_LOCK = threading.Lock()


def _words(number: int, language: str) -> str:
    with _NUM2WORDS_LOCK:
        return str(num2words(number, lang=language))


def _cue_context(text: str, end: int) -> tuple[str, int]:
    """Text before *end* without vowel marks, and the position where cue search starts.

    One extra character before that position lets a cue's ``(?<!\\w)`` see what precedes it.
    """
    context = re.sub(_MARKS, "", text[max(0, end - 2 * _CUE_WINDOW) : end])
    return context, max(0, len(context) - _CUE_WINDOW)


def _sub_numbers(pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str], text: str) -> str:
    """``pattern.sub`` where an attached clitic before digits counts as a word boundary.

    Clitics are masked with NUL (same length, so offsets hold) for matching only;
    unmatched text, including the clitic itself, is copied from the original.
    """
    masked = _CLITIC.sub(lambda m: "\0" * len(m[0]), text)
    parts: list[str] = []
    end = 0
    for match in pattern.finditer(masked):
        parts += [text[end : match.start()], repl(match)]
        end = match.end()
    return "".join(parts) + text[end:]


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
        return " ".join(_words(int(d), language) for d in cleaned if d.isdecimal())

    sign = ""
    if cleaned[0] in "+-−":
        if cleaned[0] in "-−":
            sign = "سالب " if language == "ar" else "minus "
        else:
            sign = "موجب " if language == "ar" else "plus "
        cleaned = cleaned[1:]
    integer, point, fraction = cleaned.partition(".")
    words = _words(int(integer), language)
    if language == "ar":
        words = _REPEATED_SCALE.sub(
            lambda m: f"{'مئتا' if m[1] == 'مئتان' else m[1]} {m[2]} و {m[2]}", words
        )
    if point:
        separator = " فاصلة " if language == "ar" else " point "
        words += separator + " ".join(_words(int(d), language) for d in fraction)
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
            number.startswith(("+", "0"))  # a leading zero is never part of a quantity
            or _PHONE_CUE.search(*_cue_context(text, match.start()))
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

    return _sub_numbers(_NUMBER, replace, text)


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
