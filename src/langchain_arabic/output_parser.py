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

from langchain_arabic.diacritics import apply_diacritics
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
    diacritics_map: Dict[str, str] = Field(default_factory=dict)
    convert_numbers: bool = True
    language: str = "ar"
    number_contexts: Optional[Set[str]] = None

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
        return self

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def parse(self, text: str) -> str:
        """Process a complete LLM output string."""
        if not text:
            return text

        # Step 1: Diacritization
        if self.backend == "catt":
            text = _get_catt_backend(self.catt_model).tashkeel(text)
            # Dictionary overrides applied AFTER auto-diacritization
            if self.diacritics_map:
                text = apply_diacritics(text, self.diacritics_map)
        elif self.diacritics_map:
            text = apply_diacritics(text, self.diacritics_map)

        # Step 2: Number conversion
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
