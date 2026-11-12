"""Run the optional graph with real LangGraph and a fake model."""

import copy
import runpy
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "langgraph_speech.py"))


@pytest.mark.parametrize("date", ["2026-11-10", "03/04/2026", "2026-02-29", "غدا", ""])
def test_graph_preserves_source_and_routes_retained_dates(date):
    graph = example["build_graph"]()
    source = {"facts": {"order_id": "001234", "date": date}}
    original = copy.deepcopy(source)
    result = graph.invoke(source)
    assert source == original
    assert result["facts"] == original["facts"]
    assert result["display_text"] == f"رقم الطلب 001234. الموعد {date}."
    assert graph.invoke(source) == result
    if date == "2026-11-10":
        assert result["status"] == "ready"
        assert "صفر صفر واحد اثنان ثلاثة أربعة" in result["speech_text"]
        assert "العاشر من نوفمبر" in result["speech_text"]
    else:
        assert result["status"] == "needs_clarification"
        assert result["clarification"] and result["speech_text"] == ""
