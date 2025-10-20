"""Execute Python code blocks from a Markdown file as isolated tests.

Each fenced ```python block becomes its own unittest test case. The script
compiles and executes the code in a fresh namespace so examples can be
validated independently.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import types
import unittest
from typing import Iterable


def extract_python_blocks(markdown_path: pathlib.Path) -> list[tuple[int, str]]:
    """Return a list of (start_line, code) for ```python code fences."""

    blocks: list[tuple[int, str]] = []
    in_block = False
    fence_lang: str | None = None
    buffer: list[str] = []
    start_line = 0

    with markdown_path.open("r", encoding="utf-8") as md_file:
        for lineno, line in enumerate(md_file, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                lang = stripped[3:].strip()
                if not in_block:
                    fence_lang = lang.lower()
                    if fence_lang.startswith("python"):
                        in_block = True
                        buffer = []
                        start_line = lineno + 1
                    else:
                        fence_lang = None
                else:
                    if in_block and fence_lang and fence_lang.startswith("python"):
                        code = "".join(buffer)
                        blocks.append((start_line, code))
                    in_block = False
                    fence_lang = None
            elif in_block:
                buffer.append(line)

    return blocks


def build_test_case(markdown_path: pathlib.Path, blocks: Iterable[tuple[int, str]]) -> type[unittest.TestCase]:
    """Create a unittest.TestCase subclass with one method per code block."""

    class MarkdownExamplesTest(unittest.TestCase):
        """Dynamically generated tests for Markdown code fences."""

        maxDiff = None  # pragma: no cover - helpful for assertion diff output

    for idx, (start_line, code) in enumerate(blocks, start=1):
        test_name = f"test_block_{idx:02d}_line_{start_line}"
        compiled_code = compile(
            code,
            filename=f"{markdown_path}#L{start_line}",
            mode="exec",
        )

        module_name = f"markdown_example_{idx:02d}"

        def _make_test(compiled: types.CodeType, name: str) -> types.FunctionType:
            def _test(self: MarkdownExamplesTest) -> None:  # type: ignore[name-defined]
                module = types.ModuleType(name)
                module.__dict__["__builtins__"] = __builtins__
                sys.modules[name] = module
                try:
                    exec(compiled, module.__dict__)  # noqa: S102 - deliberate dynamic exec
                finally:
                    sys.modules.pop(name, None)

            return _test

        setattr(MarkdownExamplesTest, test_name, _make_test(compiled_code, module_name))

    return MarkdownExamplesTest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "markdown_file",
        type=pathlib.Path,
        help="Path to the Markdown file containing code examples.",
    )
    args = parser.parse_args()

    markdown_path = args.markdown_file.resolve()
    if not markdown_path.exists():
        parser.error(f"Markdown file not found: {markdown_path}")

    blocks = extract_python_blocks(markdown_path)
    if not blocks:
        parser.error(
            textwrap.dedent(
                f"""
                No ```python code fences found in {markdown_path}.
                """
            ).strip()
        )

    test_case = build_test_case(markdown_path, blocks)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
