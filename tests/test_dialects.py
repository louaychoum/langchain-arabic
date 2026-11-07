"""Tests for langchain_arabic.dialects module."""

import pytest

from langchain_arabic.dialects import (
    SUPPORTED_DIALECTS,
    apply_dialect_number_overrides,
    resolve_dialect_diacritics_map,
)

# ---------------------------------------------------------------------------
# SUPPORTED_DIALECTS
# ---------------------------------------------------------------------------


class TestSupportedDialects:
    def test_msa_is_supported(self):
        assert "msa" in SUPPORTED_DIALECTS

    def test_gulf_is_supported(self):
        assert "gulf" in SUPPORTED_DIALECTS

    def test_is_frozenset(self):
        assert isinstance(SUPPORTED_DIALECTS, frozenset)


# ---------------------------------------------------------------------------
# apply_dialect_number_overrides
# ---------------------------------------------------------------------------


class TestApplyDialectNumberOverrides:
    def test_msa_returns_unchanged(self):
        text = "مائة بالمائة"
        assert apply_dialect_number_overrides(text, "msa") == text

    def test_gulf_replaces_miaa(self):
        assert "مية" in apply_dialect_number_overrides("مائة", "gulf")
        assert "مائة" not in apply_dialect_number_overrides("مائة", "gulf")

    def test_gulf_replaces_percentage(self):
        result = apply_dialect_number_overrides("خمسة و تسعون بالمائة", "gulf")
        assert "بالمية" in result
        assert "بالمائة" not in result

    def test_gulf_replaces_two_hundred(self):
        result = apply_dialect_number_overrides("مئتان", "gulf")
        assert "ميتين" in result

    def test_empty_text(self):
        assert apply_dialect_number_overrides("", "gulf") == ""

    def test_no_matching_words(self):
        text = "خمسة كتب"
        assert apply_dialect_number_overrides(text, "gulf") == text

    def test_invalid_dialect_raises(self):
        with pytest.raises(ValueError, match="Unsupported dialect"):
            apply_dialect_number_overrides("text", "invalid")


# ---------------------------------------------------------------------------
# resolve_dialect_diacritics_map
# ---------------------------------------------------------------------------


class TestResolveDialectDiacriticsMap:
    def test_flat_map_returned_as_is(self):
        flat = {"تقنية": "تِقْنِيَة"}
        result = resolve_dialect_diacritics_map(flat, "msa")
        assert result == flat

    def test_flat_map_ignores_dialect(self):
        flat = {"تقنية": "تِقْنِيَة"}
        result = resolve_dialect_diacritics_map(flat, "gulf")
        assert result == flat

    def test_dialect_keyed_selects_correct_submap(self):
        keyed = {
            "msa": {"تقنية": "تِقْنِيَة"},
            "gulf": {"تقنية": "تَقْنِيَّة"},
        }
        result = resolve_dialect_diacritics_map(keyed, "gulf")
        assert result == {"تقنية": "تَقْنِيَّة"}

    def test_dialect_keyed_selects_msa(self):
        keyed = {
            "msa": {"تقنية": "تِقْنِيَة"},
            "gulf": {"تقنية": "تَقْنِيَّة"},
        }
        result = resolve_dialect_diacritics_map(keyed, "msa")
        assert result == {"تقنية": "تِقْنِيَة"}

    def test_dialect_keyed_missing_dialect_falls_back_to_msa(self):
        keyed = {
            "msa": {"تقنية": "تِقْنِيَة"},
        }
        result = resolve_dialect_diacritics_map(keyed, "gulf")
        assert result == {"تقنية": "تِقْنِيَة"}

    def test_dialect_keyed_missing_all_returns_empty(self):
        keyed = {
            "egyptian": {"تقنية": "تِقْنِيَة"},
        }
        result = resolve_dialect_diacritics_map(keyed, "gulf")
        assert result == {}

    def test_empty_map(self):
        result = resolve_dialect_diacritics_map({}, "gulf")
        assert result == {}

    def test_mixed_types_raises(self):
        mixed = {"msa": {"تقنية": "تِقْنِيَة"}, "extra": "plain_string"}
        with pytest.raises(ValueError, match="mixed types"):
            resolve_dialect_diacritics_map(mixed, "msa")

    def test_invalid_dialect_raises(self):
        with pytest.raises(ValueError, match="Unsupported dialect"):
            resolve_dialect_diacritics_map({}, "invalid")
