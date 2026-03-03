"""Optional CATT auto-diacritization backend.

Wraps ``catt-tashkeel`` (a character-level Arabic tashkeel transformer) behind
a lazy-import boundary so that users who only need dictionary mode never pull
in PyTorch.

Install with::

    pip install langchain-arabic[catt]
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — catt_tashkeel is an optional dependency
# ---------------------------------------------------------------------------
_CATT_IMPORT_ERROR: Optional[ImportError] = None

try:
    from catt_tashkeel import CATTEncoderDecoder, CATTEncoderOnly
except ImportError as _exc:
    _CATT_IMPORT_ERROR = _exc
    CATTEncoderOnly = None  # type: ignore[assignment,misc]
    CATTEncoderDecoder = None  # type: ignore[assignment,misc]


def _check_catt_installed() -> None:
    """Raise a helpful error if *catt-tashkeel* is not installed."""
    if _CATT_IMPORT_ERROR is not None:
        raise ImportError(
            "catt-tashkeel is required for the 'catt' backend. "
            "Install it with:\n\n"
            "    pip install langchain-arabic[catt]\n\n"
            f"Original error: {_CATT_IMPORT_ERROR}"
        ) from _CATT_IMPORT_ERROR


_VALID_MODELS = {"encoder_only", "encoder_decoder"}


class CATTBackend:
    """Thin wrapper around *catt-tashkeel* models.

    The underlying transformer is **not** loaded until the first call to
    :meth:`tashkeel` or :meth:`tashkeel_batch`, keeping import time cheap.

    Parameters
    ----------
    model
        ``"encoder_only"`` (faster, lower accuracy) or
        ``"encoder_decoder"`` (slower, higher accuracy).
    """

    def __init__(self, model: str = "encoder_only") -> None:
        _check_catt_installed()
        if model not in _VALID_MODELS:
            raise ValueError(
                f"Unknown CATT model {model!r}. "
                f"Choose from: {sorted(_VALID_MODELS)}"
            )
        self._model_name = model
        self._model: Optional[object] = None

    def _load_model(self) -> object:
        """Lazy-load the CATT model on first use."""
        if self._model is None:
            logger.info(
                "Loading CATT %s model (first time only — may download weights)...",
                self._model_name,
            )
            if self._model_name == "encoder_only":
                self._model = CATTEncoderOnly()
            else:
                self._model = CATTEncoderDecoder()
        return self._model

    @property
    def model_name(self) -> str:
        """The configured model variant."""
        return self._model_name

    def tashkeel(self, text: str) -> str:
        """Auto-diacritize a single text string."""
        if not text or not text.strip():
            return text
        model = self._load_model()
        return model.do_tashkeel(text)  # type: ignore[union-attr]

    def tashkeel_batch(self, texts: List[str]) -> List[str]:
        """Auto-diacritize a batch of text strings."""
        if not texts:
            return []
        model = self._load_model()
        return model.do_tashkeel_batch(texts)  # type: ignore[union-attr]
