"""Tests for langchain_arabic.diacritics module."""

import os
import tempfile

import pytest

from langchain_arabic.diacritics import (
    DiacriticsProcessor,
    apply_diacritics,
    parse_diacritics_map,
)


# ---------------------------------------------------------------------------
# parse_diacritics_map
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """\
# Persona Header

Some intro text about the persona.

## Diacritics Table
- تقنية → تِقْنِيَة
- شركة → شَرِكَة
- علم الحاسوب → عِلْمُ الحَاسُوبِ
- علم → عِلْمُ

## Routing (should be ignored)
→ Route to section A
Some → unrelated → arrows in text
"""


class TestParseDiacriticsMap:
    def test_parses_arrow_mappings(self):
        result = parse_diacritics_map(SAMPLE_MARKDOWN)
        assert result["تقنية"] == "تِقْنِيَة"
        assert result["شركة"] == "شَرِكَة"
        assert result["علم الحاسوب"] == "عِلْمُ الحَاسُوبِ"
        assert result["علم"] == "عِلْمُ"

    def test_ignores_non_mapping_lines(self):
        result = parse_diacritics_map(SAMPLE_MARKDOWN)
        assert len(result) == 4

    def test_empty_input(self):
        assert parse_diacritics_map("") == {}

    def test_no_mappings_found(self):
        assert parse_diacritics_map("Just some regular text.") == {}

    def test_dict_passthrough(self):
        mapping = {"hello": "world"}
        result = parse_diacritics_map(mapping)
        assert result == {"hello": "world"}
        # Should be a copy, not the same object
        assert result is not mapping

    def test_file_path_input(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("- تقنية → تِقْنِيَة\n- شركة → شَرِكَة\n")
            path = f.name

        try:
            result = parse_diacritics_map(path)
            assert len(result) == 2
            assert result["تقنية"] == "تِقْنِيَة"
        finally:
            os.unlink(path)

    def test_skips_identical_plain_and_diacritized(self):
        md = "- same → same\n- مختلف → مُخْتَلِف\n"
        result = parse_diacritics_map(md)
        assert "same" not in result
        assert result["مختلف"] == "مُخْتَلِف"


# ---------------------------------------------------------------------------
# apply_diacritics
# ---------------------------------------------------------------------------


class TestApplyDiacritics:
    def test_longest_match_first(self):
        dmap = {
            "علم الحاسوب": "عِلْمُ الحَاسُوبِ",
            "علم": "عِلْمُ",
        }
        result = apply_diacritics("دراسة علم الحاسوب", dmap)
        assert "عِلْمُ الحَاسُوبِ" in result

    def test_multiple_replacements(self):
        dmap = {"تقنية": "تِقْنِيَة", "شركة": "شَرِكَة"}
        result = apply_diacritics("شركة تقنية هي شركة", dmap)
        assert result == "شَرِكَة تِقْنِيَة هي شَرِكَة"

    def test_empty_map_returns_text(self):
        assert apply_diacritics("hello", {}) == "hello"

    def test_empty_text_returns_empty(self):
        assert apply_diacritics("", {"a": "b"}) == ""

    def test_no_matches_returns_text(self):
        dmap = {"xyz": "XYZ"}
        assert apply_diacritics("hello world", dmap) == "hello world"


# ---------------------------------------------------------------------------
# DiacriticsProcessor
# ---------------------------------------------------------------------------


class TestDiacriticsProcessor:
    def test_process_applies_diacritics(self):
        proc = DiacriticsProcessor({"تقنية": "تِقْنِيَة"})
        assert proc.process("تقنية") == "تِقْنِيَة"

    def test_mapping_property_returns_copy(self):
        source = {"a": "b"}
        proc = DiacriticsProcessor(source)
        mapping = proc.mapping
        assert mapping == {"a": "b"}
        mapping["c"] = "d"
        assert "c" not in proc.mapping

    def test_from_markdown(self):
        proc = DiacriticsProcessor(SAMPLE_MARKDOWN)
        assert len(proc.mapping) == 4
        assert proc.process("شركة تقنية") == "شَرِكَة تِقْنِيَة"
