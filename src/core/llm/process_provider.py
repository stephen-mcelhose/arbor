"""Process-based LLM provider — BYOA (bring your own agent).

Routes every LLM call to an external command over stdin/stdout so any agent
(Claude Code, Codex, a local model server, …) can drive a native ``arbor run``
without an API key.

Protocol
--------
For each completion request ``arbor`` writes one JSON object to the command's
**stdin** and reads one JSON object from **stdout**:

Input (sent to the agent command):

.. code-block:: json

    {
        "system": "<system prompt>",
        "messages": [{"role": "user|assistant", "content": ...}, ...],
        "tools": [{"name": "...", "description": "...", "input_schema": {...}}, ...],
        "max_tokens": 16384
    }

Output (the agent command must write to stdout):

.. code-block:: json

    {
        "stop_reason": "end_turn | tool_use | max_tokens",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "toolu_...", "name": "tool_name",
             "input": {...}}
        ]
    }

Usage
-----
In ``research_config.yaml`` (or ``~/.arbor/config.yaml``):

.. code-block:: yaml

    llm:
      provider: process
      agent_cmd: /path/to/my-agent-bridge
      model: my-agent          # informational label only

The bridge is re-invoked for every LLM turn so it can be a simple stateless
script. ``arbor`` passes the full conversation history in ``messages`` each
time, so the bridge never needs to track state.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

from .base import (
    LLMProvider,
    LLMResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
    make_tool_use_id,
)


class ProcessProvider(LLMProvider):
    """LLM provider that delegates to an external agent command."""

    def __init__(self, *, agent_cmd: str, model: str = "process-agent") -> None:
        if not agent_cmd:
            raise ValueError(
                "provider=process requires agent_cmd — set it in research_config.yaml:\n"
                "  llm:\n"
                "    provider: process\n"
                "    agent_cmd: /path/to/my-agent-bridge\n"
            )
        self.agent_cmd = agent_cmd
        self.model = model

    async def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 16384,
    ) -> LLMResponse:
        payload = json.dumps({
            "system": system,
            "messages": messages,
            "tools": tools or [],
            "max_tokens": max_tokens,
        })

        args = shlex.split(self.agent_cmd)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(payload.encode())

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"agent_cmd exited with code {proc.returncode}: {err}"
            )

        try:
            raw = json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"agent_cmd returned invalid JSON: {exc}\nstdout: {stdout[:200]!r}"
            ) from exc

        content = _parse_content(raw.get("content", []))
        stop_reason = raw.get("stop_reason", "end_turn")

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            usage=Usage(),
            model=self.model,
            raw_content=raw.get("content", []),
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _parse_content(blocks: list[dict[str, Any]]) -> list:
    out = []
    for block in blocks:
        t = block.get("type", "text")
        if t == "text":
            out.append(TextBlock(text=block.get("text", "")))
        elif t == "tool_use":
            out.append(ToolUseBlock(
                id=block.get("id") or make_tool_use_id(),
                name=block.get("name", ""),
                input=block.get("input", {}),
            ))
        # ignore unknown block types — be lenient with agent output
    return out
