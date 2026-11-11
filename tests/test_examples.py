"""Keep the notebook examples runnable without keys or notebook dependencies."""

import ast
import inspect
import json
from pathlib import Path

import pytest

from langchain_arabic import __version__


@pytest.mark.asyncio
async def test_notebook_cells():
    notebook = Path(__file__).parents[1] / "examples" / "quickstart.ipynb"
    namespace = {}
    install_cells = 0
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            assert source.strip() == f'%pip -q install "langchain-arabic=={__version__}"'
            install_cells += (
                1  # Test the local package; PyPI publication is a separate release action.
            )
            continue
        result = eval(
            compile(source, str(notebook), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace
        )
        if inspect.isawaitable(result):
            await result
    assert install_cells == 1
