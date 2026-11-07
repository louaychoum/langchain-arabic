"""Arabic and English number-to-word conversion for LLM output text.

Detects numbers in text, classifies their context (year, percentage,
currency, phone, plain), and converts them to words using ``num2words``.
"""

from __future__ import annotations

import re

from num2words import num2words

_PATTERNS: list[tuple] = [
    # Percentage: 95% or 95 %
    (re.compile(r"(\d[\d,]*\.?\d*)\s*[%٪]"), "percentage", 1),
    # Currency (Arabic): 500 ريال / درهم / دينار
    (re.compile(r"(\d[\d,]*\.?\d*)\s*(ريال|درهم|دينار)"), "currency_ar", 1),
    # Currency (English): SAR 500 or 500 SAR/USD/AED
    (re.compile(r"(?:SAR|USD|AED|QAR)\s*(\d[\d,]*\.?\d*)"), "currency_en", 1),
    (re.compile(r"(\d[\d,]*\.?\d*)\s*(?:SAR|USD|AED|QAR)"), "currency_en", 1),
    # Phone: 7+ consecutive digits (or starting with + or 920/800)
    (re.compile(r"\+?\d{7,15}"), "phone", 0),
    (re.compile(r"(?:920|800)\d{6}"), "phone", 0),
]

_STANDALONE_NUMBER = re.compile(r"\d[\d,]*\.?\d*|\d")


def _convert_single(number_str: str, context: str, language: str) -> str:
    """Convert a single number string to words based on context."""
    cleaned = number_str.replace(",", "")

    if context == "phone":
        return " ".join(
            num2words(int(d), lang=language) for d in cleaned if d.isdigit()
        )

    try:
        value = float(cleaned) if "." in cleaned else int(cleaned)
    except ValueError:
        return number_str

    words = num2words(value, lang=language)

    words = re.sub(r" {2,}", " ", words)

    if context == "percentage":
        suffix = "بالمائة" if language == "ar" else "percent"
        return f"{words} {suffix}"

    return words


def convert_numbers_in_text(
    text: str,
    language: str = "ar",
    *,
    contexts: set[str] | None = None,
) -> str:
    """Replace digit sequences in *text* with their word equivalents.

    Parameters
    ----------
    text
        Input text possibly containing digits.
    language
        Target language for conversion (``"ar"`` or ``"en"``).
    contexts
        Optional set of contexts to enable. Defaults to all.
        Valid values: ``"percentage"``, ``"currency_ar"``,
        ``"currency_en"``, ``"phone"``, ``"plain"``.

    Returns
    -------
    str
        Text with numbers replaced by words.
    """
    if not text:
        return text

    all_contexts = contexts is None

    for pattern, ctx, group_idx in _PATTERNS:
        if not all_contexts and ctx not in contexts:
            continue

        def _replacer(m: re.Match, _ctx: str = ctx, _gi: int = group_idx) -> str:
            full = m.group(0)
            num_str = m.group(_gi) if _gi > 0 else full

            # Strip leading + from phone
            digits_only = num_str.lstrip("+")
            converted = _convert_single(digits_only, _ctx, language)

            # For percentage, the conversion already includes "بالمائة"/"percent"
            # so replace the entire match (including the % sign)
            if _ctx == "percentage":
                return converted
            if _gi > 0:
                return full.replace(num_str, converted)
            return converted

        text = pattern.sub(_replacer, text)

    if all_contexts or "plain" in contexts:

        def _plain_replacer(m: re.Match) -> str:
            return _convert_single(m.group(0), "plain", language)

        text = _STANDALONE_NUMBER.sub(_plain_replacer, text)

    return text


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
        contexts: set[str] | None = None,
    ) -> None:
        self._language = language
        self._contexts = contexts

    @property
    def language(self) -> str:
        return self._language

    def process(self, text: str) -> str:
        """Convert numbers in *text* to words."""
        return convert_numbers_in_text(text, self._language, contexts=self._contexts)
