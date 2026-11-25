"""Arabic TTS -> ASR round trip: raw LLM text vs this package's speech text.

Each case is synthesized by a TTS vendor, transcribed by an ASR asked for digits,
and scored by whether every expected number (and listed word) survives. Digits
are compared, not number words, because Arabic number words vary with gender and
case. ASR mistakes look like TTS mistakes: compare two ASRs and listen to failures.

Paid APIs; keys come from the environment. ``--dry-run`` checks the pipeline
offline and its scores mean nothing. Audio and transcripts are cached in --out by
text, so reruns do not pay twice; use a new --out after changing voice or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Dict, List
from xml.sax.saxutils import escape

from langchain_arabic import ArabicSpeechNormalizer

_DATA = Path(__file__).with_name("data") / "tts_roundtrip.json"
_DEFAULT_OUT = Path(__file__).with_name("tts-runs") / "latest"
_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٪−", "01234567890123456789.%-")
_MARKS = re.compile(r"[ً-ْٰ]")
_SCALES = {"ألف": 10**3, "آلاف": 10**3, "مليون": 10**6, "ملايين": 10**6}
_SCALE_TERM = r"(?:(\d+(?:\.\d+)?)\s*)?(آلاف|ألف|ملايين|مليون)ا?(?![^\W\d_])"
# "101 ألف", "مليون و250 ألف": digits with scale words, summed. Other word forms miss.
_SCALED = re.compile(rf"(?<![^\W\d_]){_SCALE_TERM}(?:\s+و\s*{_SCALE_TERM})*")
# "49 دولار و99 سنت" is 49.99.
_SUBUNIT = re.compile(r"(\d+)\s*\S+\s+و\s*(\d{1,2})\s*(?:هلل|سنت|فلس)")
# Digit groups split by one space or hyphen: "050 123 4567", "050-123-4567".
_DIGIT_RUN = re.compile(r"(?<![\d.])\d+(?:[ \u00a0-]\d+)+(?![.\d])")


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set the {name} environment variable for this provider.")
    return value


def _post(url: str, headers: Dict[str, str], body: bytes) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:  # report the status, never the request headers
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise RuntimeError(f"{url.split('?')[0]} returned HTTP {exc.code}: {detail}") from None


def _multipart(fields: Dict[str, str], audio: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def tts_elevenlabs(text: str) -> bytes:
    """ElevenLabs TTS as 16 kHz WAV; vendor text normalization left at its default."""
    voice = _env("ELEVENLABS_VOICE_ID")  # choose an Arabic voice from the Voice Library
    model = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
    return _post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=wav_16000",
        {"xi-api-key": _env("ELEVENLABS_API_KEY"), "Content-Type": "application/json"},
        json.dumps({"text": text, "model_id": model, "seed": 1}).encode(),
    )


def tts_azure(text: str) -> bytes:
    """Azure Speech TTS as 16 kHz mono WAV."""
    voice = os.environ.get("AZURE_TTS_VOICE", "ar-SA-HamedNeural")
    ssml = (
        f"<speak version='1.0' xml:lang='{voice[:5]}'>"
        f"<voice name='{voice}'>{escape(text)}</voice></speak>"
    )
    return _post(
        f"https://{_env('AZURE_SPEECH_REGION')}.tts.speech.microsoft.com/cognitiveservices/v1",
        {
            "Ocp-Apim-Subscription-Key": _env("AZURE_SPEECH_KEY"),
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            "User-Agent": "langchain-arabic-benchmark",
        },
        ssml.encode("utf-8"),
    )


def asr_scribe(audio: bytes) -> str:
    """ElevenLabs Scribe v3, asked to write numbers as digits."""
    fields = {"model_id": "scribe_v3", "language_code": "ar", "numbers_format": "digits"}
    body, content_type = _multipart(fields, audio)
    headers = {"xi-api-key": _env("ELEVENLABS_API_KEY"), "Content-Type": content_type}
    response = _post("https://api.elevenlabs.io/v1/speech-to-text", headers, body)
    return str(json.loads(response)["text"])


def asr_azure(audio: bytes) -> str:
    """Azure short-audio STT; its ITN form writes numbers as digits (16 kHz WAV only)."""
    language = os.environ.get("AZURE_STT_LANGUAGE", "ar-SA")
    url = (
        _env("AZURE_SPEECH_ENDPOINT").rstrip("/")
        + "/stt/speech/recognition/conversation/cognitiveservices/v1"
        + f"?language={language}&format=detailed&profanity=raw"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": _env("AZURE_SPEECH_KEY"),
        "Accept": "application/json",
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
    }
    result = json.loads(_post(url, headers, audio))
    status = result.get("RecognitionStatus")
    if status in (None, "Error"):  # transient: raise so nothing is cached
        raise RuntimeError(f"Azure STT returned RecognitionStatus={status}")
    if status != "Success":  # e.g. NoMatch: cached and visible in results.json
        return f"<{status}>"
    best = (result.get("NBest") or [{}])[0]
    return str(best.get("ITN") or result.get("DisplayText", ""))


TTS: Dict[str, Callable[[str], bytes]] = {
    "elevenlabs": tts_elevenlabs,
    "azure": tts_azure,
    "echo": lambda text: text.encode("utf-8"),  # dry run
}
ASR: Dict[str, Callable[[bytes], str]] = {
    "scribe": asr_scribe,
    "azure": asr_azure,
    "echo": lambda audio: audio.decode("utf-8"),  # dry run
}


def _fold(text: str) -> str:
    """Eastern digits and signs to ASCII, vowel marks removed, "و " attached."""
    return re.sub(r"(?<!\S)و\s+", "و", _MARKS.sub("", text.translate(_TO_ASCII)))


def _scaled_sum(match: re.Match[str]) -> str:
    terms = re.findall(_SCALE_TERM, match[0])
    return str(sum((Decimal(n) if n else 1) * _SCALES[word] for n, word in terms))


def transcript_numbers(transcript: str) -> List[str]:
    """Candidate numbers in an ASR transcript, read every plausible way.

    Each number as written, plus joined digit groups (phones and IDs), digits
    with scale words, and amounts split into a main unit and a subunit.
    """
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}(?!\d))", "", _fold(transcript))  # 1,250,000
    found = [f"{a}.{b:0>2}" for a, b in _SUBUNIT.findall(text)]
    text = _SCALED.sub(_scaled_sum, text)
    found += re.findall(r"\d+(?:\.\d+)?", text)
    for run in _DIGIT_RUN.finditer(text):
        groups = re.split(r"[ \u00a0-]", run[0])
        found += [
            "".join(groups[i:j]) for i in range(len(groups)) for j in range(i + 2, len(groups) + 1)
        ]
    return found


def _same_number(expected: str, found: str) -> bool:
    if expected.startswith("0") and len(expected) > 1 and "." not in expected:
        return expected == found  # identifiers and phones: every digit, zeros included
    try:
        return Decimal(expected) == Decimal(found)
    except InvalidOperation:
        return False


def entity_found(expected: str, transcript: str) -> bool:
    """True if any ``|`` alternative is in the transcript (numbers by value, words as text).

    A leading ``!`` inverts the check: none of the alternatives may appear.
    """
    if expected.startswith("!"):  # a wrong reading, e.g. a date read as subtraction
        return not entity_found(expected[1:], transcript)
    numbers = transcript_numbers(transcript)
    plain = _fold(transcript)
    for option in expected.split("|"):
        if re.search(r"\d", option):
            if any(_same_number(option, number) for number in numbers):
                return True
        # an ordinal followed by عشر is a teen: التاسعة is 9, التاسعة عشرة is 19
        elif re.search(re.escape(_fold(option)) + r"(?!\s*عشر)", plain):
            return True
    return False


def run(cases: List[dict], tts_names: List[str], asr_names: List[str], out: Path) -> List[dict]:
    normalizer = ArabicSpeechNormalizer()
    results = []
    for case in cases:
        for condition, text in (("raw", case["text"]), ("normalized", None)):
            text = text if text is not None else normalizer.process(case["text"])
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            for tts in tts_names:
                audio_path = out / tts / condition / f"{case['id']}-{key}.wav"
                if not audio_path.exists():
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_path.write_bytes(TTS[tts](text))
                audio = audio_path.read_bytes()
                for asr in asr_names:
                    cached = audio_path.with_suffix(f".{asr}.txt")
                    if not cached.exists():
                        cached.write_text(ASR[asr](audio), encoding="utf-8")
                    transcript = cached.read_text(encoding="utf-8")
                    hits = [entity_found(e, transcript) for e in case["expect"]]
                    results.append(
                        {
                            "case": case["id"],
                            "condition": condition,
                            "tts": tts,
                            "asr": asr,
                            "text": text,
                            "transcript": transcript,
                            "hits": hits,
                        }
                    )
    return results


def report(results: List[dict], dry_run: bool) -> str:
    lines = [
        "# Arabic TTS round-trip results",
        "",
        "Expected numbers/words found in the ASR transcript of each clip"
        " (`!` entries: wrong readings that must be absent).",
        "Entities: individual expected values. Cases: clips with every value found.",
        "",
    ]
    if dry_run:
        lines += ["**Dry run with echo providers: wiring check only, scores mean nothing.**", ""]
    lines += [
        "| TTS | ASR | Entities raw | Entities normalized | Cases raw | Cases normalized |",
        "|---|---|---:|---:|---:|---:|",
    ]
    pairs = sorted({(r["tts"], r["asr"]) for r in results})
    for tts, asr in pairs:
        cells = []
        for unit in ("entities", "cases"):
            for condition in ("raw", "normalized"):
                rows = [
                    r
                    for r in results
                    if (r["tts"], r["asr"], r["condition"]) == (tts, asr, condition)
                ]
                if unit == "entities":
                    good, total = (
                        sum(sum(r["hits"]) for r in rows),
                        sum(len(r["hits"]) for r in rows),
                    )
                else:
                    good, total = sum(all(r["hits"]) for r in rows), len(rows)
                cells.append(f"{good}/{total}")
        lines.append(f"| {tts} | {asr} | " + " | ".join(cells) + " |")
    by_key = {(r["tts"], r["asr"], r["case"], r["condition"]): all(r["hits"]) for r in results}
    for tts, asr in pairs:
        cases = sorted({r["case"] for r in results if (r["tts"], r["asr"]) == (tts, asr)})
        lost = [
            c
            for c in cases
            if by_key[(tts, asr, c, "raw")] and not by_key[(tts, asr, c, "normalized")]
        ]
        won = [
            c
            for c in cases
            if by_key[(tts, asr, c, "normalized")] and not by_key[(tts, asr, c, "raw")]
        ]
        lines += [
            "",
            f"{tts} + {asr}: normalization fixed {len(won)} ({', '.join(won) or 'none'}); "
            f"broke {len(lost)} ({', '.join(lost) or 'none'}).",
        ]
    lines += ["", "Transcripts and inputs: `results.json` next to this report.", ""]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tts", help=f"comma-separated: {', '.join(n for n in TTS if n != 'echo')}")
    cli.add_argument("--asr", help=f"comma-separated: {', '.join(n for n in ASR if n != 'echo')}")
    cli.add_argument("--cases", help="comma-separated case ids (default: all)")
    cli.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="cache and report directory")
    cli.add_argument("--dry-run", action="store_true", help="offline echo providers")
    args = cli.parse_args(argv)
    if args.dry_run:
        tts_names, asr_names = ["echo"], ["echo"]
    elif not (args.tts and args.asr):
        cli.error("choose --tts and --asr (paid APIs), or --dry-run")
    else:  # dict.fromkeys drops repeats, which would count a pair twice
        tts_names = list(dict.fromkeys(args.tts.split(",")))
        asr_names = list(dict.fromkeys(args.asr.split(",")))
    if "echo" in tts_names + asr_names and not args.dry_run:
        cli.error("echo providers are for --dry-run only")
    unknown = [n for n in tts_names if n not in TTS] + [n for n in asr_names if n not in ASR]
    if unknown:
        cli.error(f"unknown provider(s): {', '.join(unknown)}")
    cases = json.loads(_DATA.read_text(encoding="utf-8"))
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
        if len(cases) != len(wanted):
            cli.error("unknown case id in --cases")
    results = run(cases, tts_names, asr_names, args.out)
    text = report(results, args.dry_run)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
