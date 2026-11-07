"""Optional CATT auto-diacritization backend.

Wraps ``catt-tashkeel`` (a character-level Arabic tashkeel transformer) behind
a lazy-import boundary so that users who only need dictionary mode never pull
in the neural runtime.

Install with::

    pip install langchain-arabic[catt]
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, List, Optional, Protocol

logger = logging.getLogger(__name__)

_CATT_IMPORT_ERROR: Optional[ImportError] = None

CATTEncoderOnly: Any = None
CATTEncoderDecoder: Any = None


def _check_catt_installed() -> None:
    """Raise a helpful error if *catt-tashkeel* is not installed."""
    global CATTEncoderOnly, CATTEncoderDecoder, _CATT_IMPORT_ERROR
    if _CATT_IMPORT_ERROR is None and CATTEncoderOnly is None:
        try:
            from catt_tashkeel import CATTEncoderDecoder, CATTEncoderOnly
        except ImportError as exc:
            _CATT_IMPORT_ERROR = exc
    if _CATT_IMPORT_ERROR is not None:
        raise ImportError(
            "catt-tashkeel is required for the 'catt' backend. "
            "Install it with:\n\n"
            "    pip install langchain-arabic[catt]\n\n"
            f"Original error: {_CATT_IMPORT_ERROR}"
        ) from _CATT_IMPORT_ERROR


_VALID_MODELS = {"encoder_only", "encoder_decoder"}
_ARABIC_LETTER = r"[\u0621-\u063a\u0641-\u064a]"
_VOCALIZED_LETTER = re.compile(_ARABIC_LETTER + r"[\u064b-\u0652\u0670]*")


class _TashkeelModel(Protocol):
    def do_tashkeel(self, text: str) -> str: ...
    def do_tashkeel_batch(self, texts: List[str]) -> List[str]: ...


def _restore_text(original: str, prediction: str) -> str:
    """Transfer only vowel marks, retaining original letters and formatting."""
    source = _VOCALIZED_LETTER.findall(original)
    predicted = _VOCALIZED_LETTER.findall(prediction)
    if [word[0] for word in source] != [word[0] for word in predicted]:
        raise ValueError("CATT changed Arabic base letters; refusing to replace the original text")
    replacements = iter(predicted)
    return _VOCALIZED_LETTER.sub(lambda _match: next(replacements), original)


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
            raise ValueError(f"Unknown CATT model {model!r}. Choose from: {sorted(_VALID_MODELS)}")
        self._model_name = model
        self._model: Optional[_TashkeelModel] = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> _TashkeelModel:
        """Lazy-load the CATT model on first use."""
        with self._load_lock:
            if self._model is None:
                logger.info("Loading CATT %s model (may download weights)", self._model_name)
                if self._model_name == "encoder_only":
                    self._model = CATTEncoderOnly()
                else:
                    self._model = CATTEncoderDecoder()
        assert self._model is not None
        return self._model

    @property
    def model_name(self) -> str:
        """The configured model variant."""
        return self._model_name

    def tashkeel(self, text: str) -> str:
        """Auto-diacritize a single text string."""
        if not re.search(_ARABIC_LETTER, text):
            return text
        model = self._load_model()
        return _restore_text(text, model.do_tashkeel(text))

    def tashkeel_batch(self, texts: List[str]) -> List[str]:
        """Auto-diacritize a batch of text strings."""
        indices = [i for i, text in enumerate(texts) if re.search(_ARABIC_LETTER, text)]
        if not indices:
            return list(texts)
        model = self._load_model()
        predictions = model.do_tashkeel_batch([texts[i] for i in indices])
        if len(predictions) != len(indices):
            raise ValueError("CATT returned the wrong number of batch results")
        result = list(texts)
        for index, prediction in zip(indices, predictions):
            result[index] = _restore_text(texts[index], prediction)
        return result
