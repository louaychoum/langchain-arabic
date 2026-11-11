"""Conservative, inspectable Arabic readings for structured speech input."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, Sequence

from langchain_arabic.numbers import _NUMBER, _PHONE_CUE, convert_numbers_in_text

DateOrder = Literal["dmy", "mdy"]
_DIGITS = ("صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة")
_MONTHS = (
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)
_DAYS = (
    "الأول",
    "الثاني",
    "الثالث",
    "الرابع",
    "الخامس",
    "السادس",
    "السابع",
    "الثامن",
    "التاسع",
    "العاشر",
    "الحادي عشر",
    "الثاني عشر",
    "الثالث عشر",
    "الرابع عشر",
    "الخامس عشر",
    "السادس عشر",
    "السابع عشر",
    "الثامن عشر",
    "التاسع عشر",
    "العشرون",
    "الحادي والعشرون",
    "الثاني والعشرون",
    "الثالث والعشرون",
    "الرابع والعشرون",
    "الخامس والعشرون",
    "السادس والعشرون",
    "السابع والعشرون",
    "الثامن والعشرون",
    "التاسع والعشرون",
    "الثلاثون",
    "الحادي والثلاثون",
)
_HOURS = (
    "صفر",
    "الأولى",
    "الثانية",
    "الثالثة",
    "الرابعة",
    "الخامسة",
    "السادسة",
    "السابعة",
    "الثامنة",
    "التاسعة",
    "العاشرة",
    "الحادية عشرة",
    "الثانية عشرة",
    "الثالثة عشرة",
    "الرابعة عشرة",
    "الخامسة عشرة",
    "السادسة عشرة",
    "السابعة عشرة",
    "الثامنة عشرة",
    "التاسعة عشرة",
    "العشرون",
    "الحادية والعشرون",
    "الثانية والعشرون",
    "الثالثة والعشرون",
)
_FEMININE = ("", "واحدة", "اثنتان", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثماني", "تسع")
_CURRENCIES = {
    "SAR": "ريال سعودي",
    "USD": "دولار أمريكي",
    "AED": "درهم إماراتي",
    "QAR": "ريال قطري",
}
_CUE = re.compile(
    r"(?<!\w)(?:رقم(?: الطلب| الحجز| الحساب)?|رمز(?: التحقق)?|معرف|"
    r"order(?: id)?|booking(?: id)?|otp|id)[ \t]*[:：#]?[ \t]*$",
    re.IGNORECASE,
)
# Earlier alternatives claim spans before the generic number rule can touch them.
_STRUCTURED = (
    r"(?P<protected>https?://[^\s<>]+|www\.[^\s<>]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|```[\s\S]*?(?:```|\Z)|`[^`\n]*`|"
    r"(?<!\w)\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?!\w)|"
    r"(?<!\w)(?=[\w./:+-]*[^\W\d_])(?=[\w./:+-]*\d)[\w]+(?:[./:+-][\w]+)*)"
    r"|(?P<date>(?<![\w./:+-])\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?![\w/:+-]|\.\d))"
    r"|(?P<time>(?<![\w./:+-])\d{1,2}:\d{2}(?::\d{2})?(?:[ \t]*[APap][Mm])?(?![\w/:+-]|\.\d))"
    r"|(?P<unknown>(?<!\w)(?:\d+(?:\.\d+)?(?:[\-–—/:]\d+(?:\.\d+)?)+|\d+(?:\.\d+){2,})(?!\w))"
)
_COMMA_NUMBER = re.compile(r"[+−-]?\d+(?:[,٬]\d+)+(?:[.٫]\d+)?")


@dataclass(frozen=True)
class SpeechChange:
    """A changed or deliberately retained span; offsets refer to the input string."""

    start: int
    end: int
    original: str
    replacement: str
    kind: str


@dataclass(frozen=True)
class SpeechResult:
    """Normalized text and ordered decisions, including protected/ambiguous spans."""

    text: str
    changes: tuple[SpeechChange, ...]


def _digits(text: str) -> str:
    return " ".join(_DIGITS[int(char)] for char in text)


def _duration(value: int, unit: str, dual: str, plural: str) -> str:
    if value == 1:
        return unit + " واحدة"
    if value == 2:
        return dual
    if value < 10:
        words = _FEMININE[value]
    elif value == 10:
        words = "عشر"
    elif value == 11:
        words = "إحدى عشرة"
    elif value == 12:
        words = "اثنتا عشرة"
    elif value < 20:
        words = _FEMININE[value - 10] + " عشرة"
    else:
        tens, ones = divmod(value, 10)
        words = convert_numbers_in_text(str(tens * 10))
        if ones:
            words = ("إحدى" if ones == 1 else _FEMININE[ones]) + " و " + words
    return words + " " + (plural if value <= 10 else unit)


class ArabicSpeechNormalizer:
    """Opt-in MSA speech formatting, independent of the legacy number converter.

    Recognizes Gregorian ISO dates, 24-hour HH:MM[:SS], decimal amounts and
    numeric identifiers. Ambiguous/invalid structures, URLs, emails, inline code
    and alphanumeric codes are retained. ``date_order`` explicitly enables
    day/month/year or month/day/year dates. ``protected_terms`` are exact literal
    spans that bypass all conversions. This does not infer dialect or noun case.
    """

    def __init__(
        self, *, date_order: DateOrder | None = None, protected_terms: Sequence[str] = ()
    ) -> None:
        if date_order not in {None, "dmy", "mdy"}:
            raise ValueError("date_order must be None, 'dmy', or 'mdy'")
        if isinstance(protected_terms, str) or any(
            not isinstance(term, str) or not term for term in protected_terms
        ):
            raise ValueError("protected_terms must contain nonempty strings")
        self._date_order = date_order
        terms = sorted(set(protected_terms), key=len, reverse=True)
        literal = "|".join(re.escape(term) for term in terms)
        self._literals = re.compile(literal) if terms else None
        pattern = (rf"(?P<literal>{literal})|" if terms else "") + _STRUCTURED
        self._pattern = re.compile(pattern + rf"|(?P<numeric>{_NUMBER.pattern})")

    def process(self, text: str) -> str:
        """Return a speech string; no models, network requests or audio are involved."""
        return self.explain(text).text

    def explain(self, text: str) -> SpeechResult:
        """Return the reading plus source-offset decisions; never rescan generated words."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        decisions: list[SpeechChange] = []
        protected = [m.span() for m in self._literals.finditer(text)] if self._literals else []
        # Unsupported comma decimals/grouping must not be split into separate quantities.
        protected.extend(
            m.span() for m in _COMMA_NUMBER.finditer(text) if not _NUMBER.fullmatch(m[0])
        )

        def replace(match: re.Match[str]) -> str:
            original = match[0]
            if any(start < match.end() and end > match.start() for start, end in protected):
                replacement, kind = original, "protected"
            elif match["date"] is not None:
                replacement = self._date(original)
                kind = "date" if replacement != original else "retained_date"
            elif match["time"] is not None:
                replacement = self._time(original)
                kind = "time" if replacement != original else "retained_time"
            elif match["numeric"] is not None:
                value = match["number"]
                prefix = (match["prefix"] or "").strip()
                suffix = (match["suffix"] or "").strip()
                cue = re.sub(r"[\u064b-\u0652\u0670]", "", text[: match.start()])
                id_cue = _CUE.search(cue)
                phone_cue = _PHONE_CUE.search(cue)
                if (id_cue or phone_cue) and not (prefix or suffix):
                    if value.isdecimal():
                        replacement = _digits(value)
                        kind = "identifier" if id_cue else "phone"
                    elif phone_cue and re.fullmatch(r"\+\d+", value):
                        replacement, kind = "زائد " + _digits(value[1:]), "phone"
                    else:
                        replacement, kind = original, "protected"
                elif (
                    value.isdecimal()
                    and len(value) > 1
                    and int(value[0]) == 0
                    and not (prefix or suffix)
                ):
                    replacement, kind = _digits(value), "identifier"
                elif prefix and suffix:
                    replacement, kind = original, "protected"
                elif prefix in _CURRENCIES or suffix in _CURRENCIES:
                    label = _CURRENCIES[prefix or suffix]
                    replacement = convert_numbers_in_text(value, contexts={"plain"}) + " " + label
                    kind = "amount"
                elif (
                    re.fullmatch(r"\+?\d{7,15}", value)
                    and not (prefix or suffix)
                    and value.startswith("+")
                ):
                    replacement = ("زائد " if value.startswith("+") else "") + _digits(
                        value.lstrip("+")
                    )
                    kind = "phone"
                else:
                    replacement = convert_numbers_in_text(original)
                    kind = "number" if replacement != original else "retained_number"
            else:
                replacement, kind = original, "protected"
            decisions.append(SpeechChange(match.start(), match.end(), original, replacement, kind))
            return replacement

        normalized = self._pattern.sub(replace, text)
        return SpeechResult(normalized, tuple(decisions))

    def _date(self, text: str) -> str:
        if "-" in text and "/" in text:
            return text
        parts = re.split(r"[-/]", text)
        if len(parts[0]) == 4:
            year, month, day = map(int, parts)
        elif self._date_order and len(parts[2]) == 4:
            first, second, year = map(int, parts)
            day, month = (first, second) if self._date_order == "dmy" else (second, first)
        else:
            return text
        try:
            date(year, month, day)
        except ValueError:
            return text
        return f"{_DAYS[day - 1]} من {_MONTHS[month - 1]} {convert_numbers_in_text(str(year))}"

    @staticmethod
    def _time(text: str) -> str:
        # shortcut: retain AM/PM forms; add a locale policy before supporting 12-hour input.
        if re.search(r"[APap][Mm]$", text):
            return text
        values = [int(part) for part in text.split(":")]
        hour, minute = values[:2]
        second = values[2] if len(values) == 3 else 0
        if hour > 23 or minute > 59 or second > 59:
            return text
        result = _HOURS[hour]
        if minute:
            result += " و " + _duration(minute, "دقيقة", "دقيقتان", "دقائق")
        if second:
            result += " و " + _duration(second, "ثانية", "ثانيتان", "ثوان")
        if not minute and not second:
            result += " تماما"
        return result
