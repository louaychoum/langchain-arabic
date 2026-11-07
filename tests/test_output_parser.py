"""Tests for langchain_arabic.output_parser module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_arabic.output_parser import ArabicTextOutputParser


class TestArabicTextOutputParser:
    def test_diacritics_only(self):
        parser = ArabicTextOutputParser(
            diacritics_map={"تقنية": "تِقْنِيَة"},
            convert_numbers=False,
        )
        result = parser.parse("شركة تقنية")
        assert result == "شركة تِقْنِيَة"

    def test_numbers_only(self):
        parser = ArabicTextOutputParser(
            diacritics_map={},
            convert_numbers=True,
            language="ar",
        )
        result = parser.parse("العدد 5")
        assert "خمسة" in result

    def test_both_diacritics_and_numbers(self):
        parser = ArabicTextOutputParser(
            diacritics_map={"شركة": "شَرِكَة"},
            convert_numbers=True,
            language="ar",
        )
        result = parser.parse("شركة رقم 5")
        assert "شَرِكَة" in result
        assert "5" not in result

    def test_invoke_works(self):
        parser = ArabicTextOutputParser(
            diacritics_map={"تقنية": "تِقْنِيَة"},
            convert_numbers=False,
        )
        result = parser.invoke("تقنية")
        assert result == "تِقْنِيَة"

    def test_type_property(self):
        parser = ArabicTextOutputParser()
        assert parser._type == "arabic_text"

    def test_english_mode(self):
        parser = ArabicTextOutputParser(
            language="en",
            convert_numbers=True,
        )
        result = parser.parse("the year 2030")
        assert "two thousand and thirty" in result

    def test_empty_input(self):
        parser = ArabicTextOutputParser(
            diacritics_map={"a": "b"},
            convert_numbers=True,
        )
        assert parser.parse("") == ""

    def test_no_processing(self):
        parser = ArabicTextOutputParser(
            diacritics_map={},
            convert_numbers=False,
        )
        text = "hello world 123"
        assert parser.parse(text) == text

    def test_string_diacritics_map_auto_parsed(self):
        """String input for diacritics_map is auto-coerced via parse_diacritics_map."""
        parser = ArabicTextOutputParser(
            diacritics_map="تقنية -> تِقْنِيَة\nشركة -> شَرِكَة",
            convert_numbers=False,
        )
        result = parser.parse("شركة تقنية")
        assert result == "شَرِكَة تِقْنِيَة"

    def test_string_diacritics_map_colon_format(self):
        parser = ArabicTextOutputParser(
            diacritics_map="تقنية: تِقْنِيَة",
            convert_numbers=False,
        )
        assert parser.parse("تقنية") == "تِقْنِيَة"

    def test_number_contexts(self):
        parser = ArabicTextOutputParser(
            convert_numbers=True,
            language="ar",
            number_contexts={"percentage"},
        )
        result = parser.parse("95% من 500 عنصر")
        assert "بالمائة" in result
        # 500 should stay as digits (only percentage context enabled)
        assert "500" in result


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------


class TestBackendValidation:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            ArabicTextOutputParser(backend="invalid")

    def test_invalid_catt_model_raises(self):
        with pytest.raises(ValueError, match="Unknown catt_model"):
            ArabicTextOutputParser(backend="catt", catt_model="invalid")

    def test_dictionary_backend_is_default(self):
        parser = ArabicTextOutputParser()
        assert parser.backend == "dictionary"

    def test_catt_model_ignored_for_dictionary(self):
        # catt_model value doesn't matter when backend is dictionary
        parser = ArabicTextOutputParser(backend="dictionary", catt_model="encoder_decoder")
        result = parser.parse("hello")
        assert result == "hello"


# ---------------------------------------------------------------------------
# CATT backend integration (mocked)
# ---------------------------------------------------------------------------


class TestCATTBackendParser:
    @pytest.fixture()
    def mock_catt(self):
        """Mock _get_catt_backend to return a controllable fake."""
        mock_backend = MagicMock()
        mock_backend.tashkeel.return_value = "نَصٌّ مُشَكَّلٌ"
        with patch(
            "langchain_arabic.output_parser._get_catt_backend",
            return_value=mock_backend,
        ):
            yield mock_backend

    def test_catt_backend_calls_tashkeel(self, mock_catt):
        parser = ArabicTextOutputParser(backend="catt", convert_numbers=False)
        result = parser.parse("نص عربي")
        mock_catt.tashkeel.assert_called_once_with("نص عربي")
        assert result == "نَصٌّ مُشَكَّلٌ"

    def test_hybrid_mode_dictionary_overrides_after_catt(self, mock_catt):
        """Dictionary overrides are applied AFTER CATT auto-diacritization."""
        mock_catt.tashkeel.return_value = "شَرِكَة تَقنِيَة"
        parser = ArabicTextOutputParser(
            backend="catt",
            diacritics_map={"تَقنِيَة": "تِقْنِيَة"},
            convert_numbers=False,
        )
        result = parser.parse("شركة تقنية")
        # CATT was called first
        mock_catt.tashkeel.assert_called_once_with("شركة تقنية")
        # Dictionary override replaced CATT's diacritization
        assert "تِقْنِيَة" in result

    def test_catt_with_numbers(self, mock_catt):
        # With CATT: numbers convert FIRST, then CATT diacritizes the result
        mock_catt.tashkeel.return_value = "العَدَد خَمْسَة"
        parser = ArabicTextOutputParser(
            backend="catt",
            convert_numbers=True,
            language="ar",
        )
        result = parser.parse("العدد 5")
        # CATT receives text with numbers already converted to words
        mock_catt.tashkeel.assert_called_once_with("العدد خمسة")
        assert "5" not in result
        assert "خَمْسَة" in result

    def test_catt_empty_input(self, mock_catt):
        parser = ArabicTextOutputParser(backend="catt", convert_numbers=False)
        assert parser.parse("") == ""
        mock_catt.tashkeel.assert_not_called()


# ---------------------------------------------------------------------------
# Streaming buffering fix
# ---------------------------------------------------------------------------


class TestStreamingBuffering:
    def test_transform_buffers_chunks(self):
        parser = ArabicTextOutputParser(
            diacritics_map={"تقنية": "تِقْنِيَة"},
            convert_numbers=False,
        )
        # Simulate LLM streaming "تقنية" as two chunks
        chunks = iter(["تقن", "ية"])
        results = list(parser._transform(chunks))
        assert len(results) == 1
        assert results[0] == "تِقْنِيَة"

    def test_transform_empty_input(self):
        parser = ArabicTextOutputParser(convert_numbers=False)
        results = list(parser._transform(iter([])))
        assert results == []

    def test_transform_single_chunk(self):
        parser = ArabicTextOutputParser(
            diacritics_map={"شركة": "شَرِكَة"},
            convert_numbers=False,
        )
        results = list(parser._transform(iter(["شركة"])))
        assert results == ["شَرِكَة"]

    def test_transform_with_message_objects(self):
        """Chunks with a .content attribute are handled correctly."""

        class FakeMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        parser = ArabicTextOutputParser(
            diacritics_map={"تقنية": "تِقْنِيَة"},
            convert_numbers=False,
        )
        chunks = iter([FakeMessage("تقن"), FakeMessage("ية")])
        results = list(parser._transform(chunks))
        assert len(results) == 1
        assert results[0] == "تِقْنِيَة"

    def test_transform_many_small_chunks(self):
        """Multi-word diacritics match after buffering many tiny chunks."""
        parser = ArabicTextOutputParser(
            diacritics_map={"علم الحاسوب": "عِلْمُ الحَاسُوبِ"},
            convert_numbers=False,
        )
        # Each character arrives separately
        text = "علم الحاسوب"
        chunks = iter(list(text))
        results = list(parser._transform(chunks))
        assert len(results) == 1
        assert results[0] == "عِلْمُ الحَاسُوبِ"


# ---------------------------------------------------------------------------
# for_tts_fixes convenience constructor
# ---------------------------------------------------------------------------


class TestForTtsFixes:
    def test_dict_input(self):
        parser = ArabicTextOutputParser.for_tts_fixes(
            fixes={"عداد": "عَدَّاد"},
        )
        assert parser.backend == "dictionary"
        assert parser.parse("عداد المياه") == "عَدَّاد المياه"

    def test_string_input(self):
        parser = ArabicTextOutputParser.for_tts_fixes(
            fixes="عداد -> عَدَّاد\nيفضل -> يُفَضَّل",
        )
        assert parser.parse("يفضل عداد") == "يُفَضَّل عَدَّاد"

    def test_numbers_enabled_by_default(self):
        parser = ArabicTextOutputParser.for_tts_fixes(fixes={})
        assert parser.convert_numbers is True
        result = parser.parse("العدد 5")
        assert "خمسة" in result

    def test_numbers_disabled(self):
        parser = ArabicTextOutputParser.for_tts_fixes(
            fixes={},
            convert_numbers=False,
        )
        assert parser.parse("العدد 5") == "العدد 5"

    def test_language_passthrough(self):
        parser = ArabicTextOutputParser.for_tts_fixes(
            fixes={},
            language="en",
        )
        result = parser.parse("the year 2030")
        assert "two thousand and thirty" in result

    def test_dialect_passthrough(self):
        parser = ArabicTextOutputParser.for_tts_fixes(
            fixes={},
            dialect="gulf",
        )
        assert parser.dialect == "gulf"


# ---------------------------------------------------------------------------
# Dialect integration
# ---------------------------------------------------------------------------


class TestDialectIntegration:
    def test_dialect_default_is_msa(self):
        parser = ArabicTextOutputParser()
        assert parser.dialect == "msa"

    def test_invalid_dialect_raises(self):
        with pytest.raises(ValueError, match="Unsupported dialect"):
            ArabicTextOutputParser(dialect="invalid")

    def test_gulf_number_overrides_percentage(self):
        parser = ArabicTextOutputParser(
            dialect="gulf",
            convert_numbers=True,
            language="ar",
            number_contexts={"percentage"},
        )
        result = parser.parse("النسبة 95%")
        assert "بالمية" in result
        assert "بالمائة" not in result

    def test_msa_number_unchanged(self):
        parser = ArabicTextOutputParser(
            dialect="msa",
            convert_numbers=True,
            language="ar",
            number_contexts={"percentage"},
        )
        result = parser.parse("النسبة 95%")
        assert "بالمائة" in result

    def test_gulf_with_dialect_keyed_diacritics_map(self):
        parser = ArabicTextOutputParser(
            dialect="gulf",
            diacritics_map={
                "msa": {"تقنية": "تِقْنِيَة"},
                "gulf": {"تقنية": "تَقْنِيَّة"},
            },
            convert_numbers=False,
        )
        result = parser.parse("شركة تقنية")
        assert "تَقْنِيَّة" in result

    def test_msa_with_dialect_keyed_diacritics_map(self):
        parser = ArabicTextOutputParser(
            dialect="msa",
            diacritics_map={
                "msa": {"تقنية": "تِقْنِيَة"},
                "gulf": {"تقنية": "تَقْنِيَّة"},
            },
            convert_numbers=False,
        )
        result = parser.parse("شركة تقنية")
        assert "تِقْنِيَة" in result

    def test_flat_map_with_gulf_dialect(self):
        """Flat diacritics map works regardless of dialect setting."""
        parser = ArabicTextOutputParser(
            dialect="gulf",
            diacritics_map={"تقنية": "تِقْنِيَة"},
            convert_numbers=False,
        )
        result = parser.parse("تقنية")
        assert result == "تِقْنِيَة"

    def test_backward_compatible_default(self):
        """Default dialect='msa' produces identical output to v0.2.0."""
        parser_new = ArabicTextOutputParser(
            dialect="msa",
            diacritics_map={"شركة": "شَرِكَة"},
            convert_numbers=True,
            language="ar",
        )
        parser_old = ArabicTextOutputParser(
            diacritics_map={"شركة": "شَرِكَة"},
            convert_numbers=True,
            language="ar",
        )
        text = "شركة رقم 100"
        assert parser_new.parse(text) == parser_old.parse(text)


def test_hybrid_uses_plain_dictionary_keys():
    with patch("langchain_arabic.output_parser._get_catt_backend") as backend:
        backend.return_value.tashkeel.return_value = "تَقْنِيَة"
        parser = ArabicTextOutputParser(backend="catt", diacritics_map={"تقنية": "تِقْنِيَة"})
        assert parser.parse("تقنية") == "تِقْنِيَة"


def test_real_lcel_stream_with_content_blocks():
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import RunnableGenerator

    def generate(_input):
        yield AIMessageChunk(content=[{"type": "text", "text": "تقن"}])
        yield AIMessageChunk(content=[{"type": "text", "text": "ية 95%"}])
        yield AIMessageChunk(content=[{"type": "reasoning", "reasoning": "private"}])

    chain = RunnableGenerator(generate) | ArabicTextOutputParser(diacritics_map={"تقنية": "تِقْنِيَة"})
    assert list(chain.stream("ignored")) == ["تِقْنِيَة خمسة و تسعون بالمائة"]


@pytest.mark.asyncio
async def test_async_lcel_and_processing_runs_off_event_loop(monkeypatch):
    import threading

    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import RunnableGenerator

    async def generate(_input):
        yield AIMessageChunk(content=[{"type": "text", "text": "95"}])
        yield AIMessageChunk(content=[{"type": "text", "text": "%"}])

    parser = ArabicTextOutputParser()
    assert await parser.ainvoke("5") == "خمسة"
    assert await parser.abatch(["5", "6"]) == ["خمسة", "ستة"]
    chain = RunnableGenerator(generate) | parser
    assert [x async for x in chain.astream("ignored")] == ["خمسة و تسعون بالمائة"]
    main_thread = threading.get_ident()
    threads = []

    def parse(self, text):
        threads.append(threading.get_ident())
        return text

    monkeypatch.setattr(ArabicTextOutputParser, "parse", parse)
    assert [x async for x in chain.astream("ignored")] == ["95%"]
    assert threads and all(thread != main_thread for thread in threads)


@pytest.mark.asyncio
async def test_empty_async_stream():
    async def empty():
        if False:
            yield ""

    assert [x async for x in ArabicTextOutputParser().atransform(empty())] == []


def test_changed_dialect_and_mapping_are_used():
    parser = ArabicTextOutputParser(
        diacritics_map={"msa": {"تقنية": "تِقْنِيَة"}, "gulf": {"تقنية": "تَقْنِيَّة"}},
        convert_numbers=False,
    )
    parser.dialect = "gulf"
    assert parser.parse("تقنية") == "تَقْنِيَّة"
    parser.diacritics_map = {"علم": "عِلْم"}
    assert parser.parse("علم") == "عِلْم"


@pytest.mark.parametrize("options", [{"language": "xx"}, {"number_contexts": {"typo"}}])
def test_invalid_number_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ArabicTextOutputParser(**options)


def test_backend_cache_keeps_each_model_once(monkeypatch):
    import langchain_arabic.output_parser as module

    monkeypatch.setattr(module, "_CATT_CACHE", {})
    with patch("langchain_arabic.catt_backend.CATTBackend") as backend:
        first = module._get_catt_backend("encoder_only")
        assert module._get_catt_backend("encoder_only") is first
        module._get_catt_backend("encoder_decoder")
        assert backend.call_count == 2
        assert backend.call_args_list[0].kwargs == {"model": "encoder_only"}
        assert backend.call_args_list[1].kwargs == {"model": "encoder_decoder"}


@pytest.mark.parametrize("cue", ["هاتف", "جوال", "اتصل على"])
def test_diacritized_phone_cue_still_selects_digit_reading(cue):
    mapping = {"هاتف": "هَاتِف", "جوال": "جَوَّال", "اتصل على": "اِتَّصِلْ عَلَى"}
    parser = ArabicTextOutputParser(diacritics_map=mapping)
    assert parser.parse(f"{cue}: 1234567") == (
        f"{mapping[cue]}: واحد اثنان ثلاثة أربعة خمسة ستة سبعة"
    )
