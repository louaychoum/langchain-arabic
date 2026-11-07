"""Tests for langchain_arabic.catt_backend module.

All tests mock catt_tashkeel — no PyTorch or GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import langchain_arabic.catt_backend as cb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_catt_installed(monkeypatch):
    """Pretend catt-tashkeel is installed by clearing the import error."""
    monkeypatch.setattr(cb, "_CATT_IMPORT_ERROR", None)
    monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock)
    monkeypatch.setattr(cb, "CATTEncoderDecoder", MagicMock)


# ---------------------------------------------------------------------------
# Import error handling
# ---------------------------------------------------------------------------


class TestCATTImportError:
    def test_check_raises_helpful_message(self):
        original = cb._CATT_IMPORT_ERROR
        cb._CATT_IMPORT_ERROR = ImportError("No module named 'catt_tashkeel'")
        try:
            with pytest.raises(ImportError, match=r"pip install langchain-arabic\[catt\]"):
                cb._check_catt_installed()
        finally:
            cb._CATT_IMPORT_ERROR = original

    def test_check_passes_when_installed(self, _mock_catt_installed):
        cb._check_catt_installed()  # should not raise


# ---------------------------------------------------------------------------
# CATTBackend construction
# ---------------------------------------------------------------------------


class TestCATTBackendInit:
    def test_invalid_model_raises(self, _mock_catt_installed):
        with pytest.raises(ValueError, match="Unknown CATT model"):
            cb.CATTBackend(model="invalid")

    def test_encoder_only_default(self, _mock_catt_installed):
        backend = cb.CATTBackend(model="encoder_only")
        assert backend.model_name == "encoder_only"

    def test_encoder_decoder(self, _mock_catt_installed):
        backend = cb.CATTBackend(model="encoder_decoder")
        assert backend.model_name == "encoder_decoder"

    def test_model_not_loaded_at_init(self, _mock_catt_installed):
        backend = cb.CATTBackend(model="encoder_only")
        assert backend._model is None


# ---------------------------------------------------------------------------
# tashkeel / tashkeel_batch
# ---------------------------------------------------------------------------


class TestCATTBackendTashkeel:
    def test_tashkeel_calls_model(self, _mock_catt_installed, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.do_tashkeel.return_value = "مُخْرَجَات"
        monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock(return_value=mock_instance))

        backend = cb.CATTBackend(model="encoder_only")
        result = backend.tashkeel("مخرجات")

        assert result == "مُخْرَجَات"
        mock_instance.do_tashkeel.assert_called_once_with("مخرجات")

    def test_tashkeel_empty_string(self, _mock_catt_installed):
        backend = cb.CATTBackend(model="encoder_only")
        assert backend.tashkeel("") == ""

    def test_tashkeel_whitespace_only(self, _mock_catt_installed):
        backend = cb.CATTBackend(model="encoder_only")
        assert backend.tashkeel("   ") == "   "

    def test_tashkeel_batch_calls_model(self, _mock_catt_installed, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.do_tashkeel_batch.return_value = ["اَ", "بِ"]
        monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock(return_value=mock_instance))

        backend = cb.CATTBackend(model="encoder_only")
        result = backend.tashkeel_batch(["ا", "ب"])

        assert result == ["اَ", "بِ"]
        mock_instance.do_tashkeel_batch.assert_called_once_with(["ا", "ب"])

    def test_tashkeel_batch_empty_list(self, _mock_catt_installed):
        backend = cb.CATTBackend(model="encoder_only")
        assert backend.tashkeel_batch([]) == []

    def test_lazy_loading_only_once(self, _mock_catt_installed, monkeypatch):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.do_tashkeel.return_value = "نَص"
        mock_cls.return_value = mock_instance
        monkeypatch.setattr(cb, "CATTEncoderOnly", mock_cls)

        backend = cb.CATTBackend(model="encoder_only")
        backend.tashkeel("نص")
        backend.tashkeel("نص")

        mock_cls.assert_called_once()  # Model created only once

    def test_encoder_decoder_model(self, _mock_catt_installed, monkeypatch):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.do_tashkeel.return_value = "نَصٌّ"
        mock_cls.return_value = mock_instance
        monkeypatch.setattr(cb, "CATTEncoderDecoder", mock_cls)

        backend = cb.CATTBackend(model="encoder_decoder")
        result = backend.tashkeel("نص")

        assert result == "نَصٌّ"
        mock_cls.assert_called_once()


def test_preserves_mixed_text_and_rtl_punctuation(_mock_catt_installed, monkeypatch):
    model = MagicMock()
    model.do_tashkeel.return_value = "نَص عَرَبِي"
    monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock(return_value=model))
    assert cb.CATTBackend().tashkeel("Hello نص، عربي! ٠١٢\n") == "Hello نَص، عَرَبِي! ٠١٢\n"


def test_rejects_changed_base_letters(_mock_catt_installed, monkeypatch):
    model = MagicMock()
    model.do_tashkeel.return_value = "شَيء"
    monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock(return_value=model))
    with pytest.raises(ValueError, match="base letters"):
        cb.CATTBackend().tashkeel("نص")


def test_batch_retains_empty_and_non_arabic_inputs(_mock_catt_installed, monkeypatch):
    model = MagicMock()
    model.do_tashkeel_batch.return_value = ["نَص"]
    monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock(return_value=model))
    assert cb.CATTBackend().tashkeel_batch(["", "English 12", "نص"]) == ["", "English 12", "نَص"]
    model.do_tashkeel_batch.assert_called_once_with(["نص"])


def test_batch_rejects_missing_results(_mock_catt_installed, monkeypatch):
    model = MagicMock()
    model.do_tashkeel_batch.return_value = []
    monkeypatch.setattr(cb, "CATTEncoderOnly", MagicMock(return_value=model))
    with pytest.raises(ValueError, match="batch results"):
        cb.CATTBackend().tashkeel_batch(["نص"])


def test_concurrent_first_calls_load_once(_mock_catt_installed, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()
    model = MagicMock()
    model.do_tashkeel.return_value = "نَص"

    def load():
        entered.set()
        assert release.wait(2)
        return model

    constructor = MagicMock(side_effect=load)
    monkeypatch.setattr(cb, "CATTEncoderOnly", constructor)
    backend = cb.CATTBackend()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.tashkeel, "نص")
        assert entered.wait(2)
        second = pool.submit(backend.tashkeel, "نص")
        release.set()
        assert first.result() == second.result() == "نَص"
    constructor.assert_called_once()


def test_dictionary_import_does_not_import_neural_runtime():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys, langchain_arabic; assert "catt_tashkeel" not in sys.modules',
        ],
        check=True,
    )
