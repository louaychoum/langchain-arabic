"""Run the key-free Pipecat example through a real Pipecat pipeline."""

import runpy
from pathlib import Path

import pytest

pytest.importorskip("pipecat")
example = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "pipecat_bot.py"))


@pytest.mark.asyncio
async def test_pipecat_pipeline_speaks_normalized_sentences():
    spoken = await example["main"]()
    assert spoken == [
        "رقم الطلب أربعة ثمانية اثنان واحد، والمبلغ مئتان و خمسون فاصلة سبعة خمسة ريال سعودي.",
        "اتصل على صفر خمسة صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة.",
    ]
