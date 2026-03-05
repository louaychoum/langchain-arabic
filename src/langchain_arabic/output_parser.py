"""LangChain-compatible output parser for Arabic text post-processing.

Wraps diacritics restoration and number-to-word conversion into a single
``Runnable`` that slots into any LCEL chain::

    chain = prompt | llm | ArabicTextOutputParser(diacritics_map={...})

Supports two diacritization backends:

- ``"dictionary"`` (default): deterministic longest-first replacement.
- ``"catt"``: neural auto-diacritization via *catt-tashkeel*, with optional
  dictionary overrides applied afterward.  Requires
  ``pip install langchain-arabic[catt]``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, Optional, Set, Union

from langchain_core.output_parsers import BaseTransformOutputParser
from pydantic import Field, model_validator

from langchain_arabic.diacritics import apply_diacritics, parse_diacritics_map
from langchain_arabic.numbers import convert_numbers_in_text

# ---------------------------------------------------------------------------
# Module-level CATT cache — shares loaded model across parser instances
# ---------------------------------------------------------------------------
_CATT_CACHE: Dict[str, Any] = {}

_VALID_BACKENDS = {"dictionary", "catt"}
_VALID_CATT_MODELS = {"encoder_only", "encoder_decoder"}


def _get_catt_backend(model: str) -> Any:
    """Return a cached :class:`CATTBackend` for *model*."""
    if model not in _CATT_CACHE:
        from langchain_arabic.catt_backend import CATTBackend

        _CATT_CACHE[model] = CATTBackend(model=model)
    return _CATT_CACHE[model]


class ArabicTextOutputParser(BaseTransformOutputParser[str]):
    """Post-process LLM output for Arabic TTS and display.

    Applies diacritization (dictionary or neural) and/or number-to-word
    conversion.

    **Streaming note:** This parser buffers all streaming chunks and
    processes the full text at the end, because diacritics and number
    conversion require complete words.  When streaming via
    ``chain.stream()``, the processed result is yielded as a single chunk
    once the LLM finishes generating.

    Parameters
    ----------
    backend
        ``"dictionary"`` (default) for deterministic replacement, or
        ``"catt"`` for neural auto-diacritization.
    catt_model
        CATT model variant: ``"encoder_only"`` (faster) or
        ``"encoder_decoder"`` (more accurate).  Ignored when *backend*
        is ``"dictionary"``.
    diacritics_map
        Plain -> diacritized word mapping.  When *backend* is ``"catt"``,
        these overrides are applied **after** auto-diacritization (useful
        for proper nouns and brand names).
    convert_numbers
        Whether to convert digit sequences to words.
    language
        Target language (``"ar"`` or ``"en"``).
    number_contexts
        Which number contexts to convert.  ``None`` = all.
    """

    backend: str = "dictionary"
    catt_model: str = "encoder_only"
    diacritics_map: Union[str, Dict[str, str]] = Field(default_factory=dict)
    convert_numbers: bool = True
    language: str = "ar"
    number_contexts: Optional[Set[str]] = None

    # Internal resolved map — set by validator, not by user
    _resolved_map: Dict[str, str] = {}

    @model_validator(mode="after")
    def _validate_config(self) -> ArabicTextOutputParser:
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. "
                f"Choose from: {sorted(_VALID_BACKENDS)}"
            )
        if self.backend == "catt" and self.catt_model not in _VALID_CATT_MODELS:
            raise ValueError(
                f"Unknown catt_model {self.catt_model!r}. "
                f"Choose from: {sorted(_VALID_CATT_MODELS)}"
            )
        # Coerce string/path → dict via auto-detection
        if isinstance(self.diacritics_map, str):
            self._resolved_map = parse_diacritics_map(self.diacritics_map)
        else:
            self._resolved_map = self.diacritics_map
        return self

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_tts_fixes(
        cls,
        fixes: Union[str, Dict[str, str]],
        convert_numbers: bool = True,
        language: str = "ar",
        **kwargs: Any,
    ) -> ArabicTextOutputParser:
        """Create a parser that only fixes specific mispronounced words.

        Recommended for TTS pipelines where only a few words need
        correction. No neural model is loaded — fast and lightweight.

        Parameters
        ----------
        fixes
            Dict, file path, or text with word mappings (any supported format).
        convert_numbers
            Whether to also convert digits to words.
        language
            ``"ar"`` or ``"en"``.
        """
        diacritics_map = (
            parse_diacritics_map(fixes) if isinstance(fixes, str) else fixes
        )
        return cls(
            backend="dictionary",
            diacritics_map=diacritics_map,
            convert_numbers=convert_numbers,
            language=language,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def parse(self, text: str) -> str:
        """Process a complete LLM output string."""
        if not text:
            return text

        if self.backend == "catt":
            # With CATT: convert numbers FIRST so the Arabic words
            # get diacritized too (CATT strips raw digits)
            if self.convert_numbers:
                text = convert_numbers_in_text(
                    text, self.language, contexts=self.number_contexts
                )
            text = _get_catt_backend(self.catt_model).tashkeel(text)
            # Dictionary overrides applied AFTER auto-diacritization
            if self._resolved_map:
                text = apply_diacritics(text, self._resolved_map)
        else:
            # With dictionary: diacritize first, then convert numbers
            if self._resolved_map:
                text = apply_diacritics(text, self._resolved_map)
            if self.convert_numbers:
                text = convert_numbers_in_text(
                    text, self.language, contexts=self.number_contexts
                )
        return text

    # ------------------------------------------------------------------
    # Streaming: buffer all chunks, then process full text
    # ------------------------------------------------------------------

    def _transform(
        self,
        input: Iterator[Union[str, Any]],
    ) -> Iterator[str]:
        """Buffer all streaming chunks, then process the full text once."""
        chunks: list[str] = []
        for chunk in input:
            if hasattr(chunk, "content"):
                chunks.append(str(chunk.content))
            else:
                chunks.append(str(chunk))
        full_text = "".join(chunks)
        if full_text:
            yield self.parse(full_text)

    async def _atransform(
        self,
        input: AsyncIterator[Union[str, Any]],
    ) -> AsyncIterator[str]:
        """Async version: buffer all chunks, process full text."""
        chunks: list[str] = []
        async for chunk in input:
            if hasattr(chunk, "content"):
                chunks.append(str(chunk.content))
            else:
                chunks.append(str(chunk))
        full_text = "".join(chunks)
        if full_text:
            yield self.parse(full_text)

    @property
    def _type(self) -> str:
        return "arabic_text"
