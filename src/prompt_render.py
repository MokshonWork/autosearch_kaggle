"""Prompt templating with clearly delimited data sections (G14).

Templates use <KEY> ... </KEY> blocks. render() fills each block with DATA and
leaves the delimiters in place; blocks with no data are removed entirely so the
model never sees an empty optional section. All data is text the agents must
treat as data, never instructions (stated in every prompt file).
"""
from __future__ import annotations

import re
from pathlib import Path

_BLOCK = re.compile(r"<([A-Z_]+)>(.*?)</\1>", re.DOTALL)


def render(template: str, sections: dict[str, str | None]) -> str:
    """Fill <KEY>...</KEY> blocks from `sections`; drop missing/empty ones."""
    def _sub(m: re.Match) -> str:
        key = m.group(1)
        data = sections.get(key)
        if data is None or str(data).strip() == "":
            return ""
        return f"<{key}>\n{data}\n</{key}>"
    out = _BLOCK.sub(_sub, template)
    return re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"


def load_prompt(prompts_dir: Path, name: str) -> str:
    return (Path(prompts_dir) / f"{name}.md").read_text()
