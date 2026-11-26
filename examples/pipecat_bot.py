"""Key-free Pipecat pipeline: Arabic numbers are rewritten before the TTS speaks them.

Install examples/requirements-pipecat.txt first (Python 3.11+), then run
``python examples/pipecat_bot.py``. PrintTTS stands in for a real service: pass the
same ``text_transforms`` to ElevenLabs, Cartesia or any other Pipecat TTS service.
Tested with pipecat-ai 1.17.0.
"""

import asyncio

from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.workers.runner import WorkerRunner

from langchain_arabic.voice import pipecat_text_transform

# An LLM reply as it streams, split mid-number the way real tokens are.
TOKENS = [
    "رقم الطلب ",
    "48",
    "21",
    "، والمبلغ ",
    "250",
    ".75",
    " SAR. ",
    "اتصل على ",
    "050",
    "1234567.",
]


class PrintTTS(TTSService):
    """Prints the text a real TTS service would receive."""

    def __init__(self, **kwargs):
        super().__init__(settings=TTSSettings(model=None, voice=None, language=None), **kwargs)
        self.spoken: list[str] = []

    async def run_tts(self, text: str, context_id: str):
        self.spoken.append(text)
        print("TTS speaks:", text)
        yield None  # a real service yields audio frames here


async def main() -> list[str]:
    tts = PrintTTS(text_transforms=[("*", pipecat_text_transform())])
    worker = PipelineWorker(Pipeline([tts]))
    await worker.queue_frames(
        [
            LLMFullResponseStartFrame(),
            *(LLMTextFrame(token) for token in TOKENS),
            LLMFullResponseEndFrame(),
            EndFrame(),
        ]
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
    return tts.spoken


if __name__ == "__main__":
    asyncio.run(main())
