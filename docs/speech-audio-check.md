# Targeted speech-input check — November 11, 2026

Three synthetic cases were checked with ElevenLabs. This is an exploratory
regression check, not a benchmark or a listener preference study. The new
normalizer avoids the two previously observed default-converter regressions;
raw ElevenLabs input already worked on all three cases.

## Method

- Voice: built-in Sarah (`EXAVITQu4vr4xnSDxMaL`), English-labeled. This is not a
  claim about a native Arabic voice or Gulf accent.
- Model: `eleven_v4`; language: `ar`; seed: `4391`; text normalization: `auto`;
  output: `mp3_44100_128`; otherwise provider voice defaults.
- One synthesis per condition. Raw date and ID clips were reused from the
  November 10 pilot after verifying the exact request text, settings, voice and
  SHA-256. Four new clips were generated: the three normalized outputs and raw
  amount. No model revision was pinned, so reuse across days is a limitation.
- Diagnostic ASR: `scribe_v2`, language hint `ara`, no keyterms or expected
  transcript supplied, no diarization or audio-event tags. ASR can hide or
  introduce pronunciation errors; it is not an independent listening judgment.
- Verified all six files decode with `ffprobe`, their SHA-256 hashes match
  metadata, and their saved requests match the intended input/settings.
- Account was free with paid overages disabled. No cost or latency comparison
  is reported. Quota accounting lagged, so no credit-use figure is inferred.

## Inputs and actual diagnostic transcripts

### Identifier

Raw: `رقم الطلب 001234.`

Normalized: `رقم الطلب صفر صفر واحد اثنان ثلاثة أربعة.`

ASR for both: `رقم الطلب 001234`

The earlier 0.3.1 default converter produced a cardinal reading whose pilot ASR
was `رقم الطلب 1234`. The new path retains the two leading zeros in the observed
transcript. Raw input already retained them too.

### Date and time

Raw: `الموعد 2026-11-10 الساعة 09:30.`

Normalized: `الموعد العاشر من نوفمبر ألفان و ستة و عشرون الساعة التاسعة و ثلاثون دقيقة.`

Raw ASR: `الموعد العاشر من نوفمبر عام 2026، الساعة التاسعة والنصف`

Normalized ASR: `الموعد العاشر من نوفمبر 2026، الساعة التاسعة وثلاثون دقيقة.`

Both transcripts express the intended date and time. The earlier 0.3.1 default
converter's pilot ASR included `شرطة 11. شرطة 10` instead of a date reading.
The new path avoids that observed behavior; preference for either new or raw
wording was not measured.

### Decimal amount

Raw: `المبلغ ١٢٫٥٠ SAR.`

Normalized: `المبلغ اثنا عشر فاصلة خمسة صفر ريال سعودي.`

ASR for both: `المبلغ 12.50 ريال سعودي`

No improvement over raw input was established.

## Reproduction and limits

Generate these exact raw and normalized inputs with the settings above, then
transcribe without providing reference text. Obtain the normalized inputs with
`ArabicSpeechNormalizer().process(raw)`. Provider output can vary despite the
seed. Local clips and per-request metadata are kept outside the package and
source distribution; they are not installed or shipped as package data.

The offline [text fixtures](../benchmarks/speech-results.md) reproduce without
provider access. These six clips support only the observations above. They do
not establish general semantic safety, pronunciation accuracy, improved
naturalness, native-listener preference, or benefit with another provider.
