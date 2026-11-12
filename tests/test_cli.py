"""Exercise the actual stdin/stdout command, including invalid input."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(payload):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    return subprocess.run(
        [sys.executable, "-m", "langchain_arabic", "normalize", "--json"],
        input=payload,
        capture_output=True,
        env=env,
        check=False,
    )


def test_json_round_trip_preserves_source_and_reports_readings():
    text = "Order رقم الطلب ٠٠١٢. المبلغ 12.50 SAR. 03/04/2026 `x=12`"
    completed = run_cli(json.dumps({"text": text}).encode())
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["display_text"] == text
    assert result["speech_text"] == (
        "Order رقم الطلب صفر صفر واحد اثنان. المبلغ اثنا عشر فاصلة خمسة صفر ريال سعودي. "
        "03/04/2026 `x=12`"
    )
    assert any("retained_date" in warning for warning in result["warnings"])
    for change in result["changes"]:
        assert text[change["start"] : change["end"]] == change["original"]


def test_date_policy_and_literal_protection():
    completed = run_cli(
        json.dumps(
            {"text": "03/04/2026 0012", "date_order": "dmy", "protected_terms": ["0012"]}
        ).encode()
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["speech_text"] == "الثالث من أبريل ألفان و ستة و عشرون 0012"


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"[]",
        b"{}",
        b'{"text": 2}',
        b'{"text":"x","other":1}',
        b'{"text":"x","date_order":[]}',
        b'{"text":"x","protected_terms":"x"}',
        b'{"text":"x","protected_terms":[null]}',
        b"\xff",
        b'{"text":"\\ud800"}',
        b" " * 65_537,
    ],
)
def test_invalid_input_has_no_json_or_traceback(payload):
    completed = run_cli(payload)
    assert completed.returncode == 2
    assert not completed.stdout
    assert b"langchain-arabic:" in completed.stderr
    assert b"Traceback" not in completed.stderr


def test_empty_text_and_range_warning_stay_machine_readable():
    assert json.loads(run_cli(b'{"text":""}').stdout)["speech_text"] == ""
    completed = run_cli(json.dumps({"text": "9" * 100}).encode())
    result = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert b"retained unchanged" not in completed.stderr
    assert result["display_text"] == result["speech_text"] == "9" * 100
    assert result["warnings"]
