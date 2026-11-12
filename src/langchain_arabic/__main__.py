"""UTF-8 JSON interface to the existing speech normalizer."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from typing import Optional, Sequence, cast

from langchain_arabic.speech import ArabicSpeechNormalizer, DateOrder

_MAX_INPUT_BYTES = 65_536


def _normalize(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) - {"text", "date_order", "protected_terms"}:
        raise ValueError("Expected an object with text, optional date_order and protected_terms")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    date_order = payload.get("date_order")
    if date_order is not None and date_order not in ("dmy", "mdy"):
        raise ValueError("date_order must be null, 'dmy' or 'mdy'")
    terms = payload.get("protected_terms", [])
    if not isinstance(terms, list) or any(not isinstance(t, str) or not t for t in terms):
        raise ValueError("protected_terms must be an array of nonempty strings")
    normalizer = ArabicSpeechNormalizer(
        date_order=cast(Optional[DateOrder], date_order), protected_terms=terms
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = normalizer.explain(text)
    notices = [str(item.message) for item in caught]
    for change in result.changes:
        if change.kind.startswith("retained_"):
            notices.append(
                f"{change.kind} at [{change.start}:{change.end}]; check the intended reading"
            )
    return {
        "display_text": text,
        "speech_text": result.text,
        "changes": [asdict(change) for change in result.changes],
        "warnings": notices,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize", help="Read one UTF-8 JSON object from stdin")
    normalize.add_argument(
        "--json", action="store_true", required=True, help="Emit one JSON object"
    )
    parser.parse_args(argv)
    try:
        raw = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
        if len(raw) > _MAX_INPUT_BYTES:
            raise ValueError("Input exceeds the 65536-byte limit")
        result = _normalize(json.loads(raw.decode("utf-8")))
        output = (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    except (ValueError, UnicodeError, RecursionError) as exc:
        # Do not echo user text or a traceback into a pipeline's diagnostics.
        message = str(exc) if type(exc) is ValueError else "Invalid UTF-8 JSON input"
        print(f"langchain-arabic: {message}", file=sys.stderr)
        return 2
    try:
        sys.stdout.buffer.write(output)
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
