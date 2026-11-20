# Changelog

## 3.4.0 (2026-11-20)

- `saef[mcp]` now pins `mcp<1.30`: mcp 1.30 with the separate `mcp-types` package fails on `import mcp`, which broke fresh installs of the MCP server.

- Added `saef.integrations.llamaindex.SaefEventHandler`, a LlamaIndex instrumentation event handler. It records one estimate per finished chat or completion call from the provider's raw response, exposes `totals()` and `reset()`, and `register()` adds it to the LlamaIndex root dispatcher. New optional extra `llamaindex`.
- Added `saef.integrations.haystack.SaefTracker`, a Haystack 2.x pass-through component to place after a Generator or ChatGenerator. It reads `model` and `usage` from reply metadata, outputs the replies and a `saef` list of estimates, and keeps running totals. New optional extra `haystack`.
- Dataset unchanged, still v3.1.0.

## 3.3.2 (2026-11-18)

- The method paper is published: Sustainable Production and Consumption 70 (2026) 320-347, doi 10.1016/j.spc.2026.11.003. `saef.citation()` now returns an `@article` entry with the DOI, and the MCP server cites the journal version.
- PyPI description and keywords now say what the package is used for, tracking the footprint of OpenAI, Anthropic, LiteLLM and LangChain calls. README opens with the response-object example.
- Dataset unchanged, still v3.1.0.

## 3.3.1 (2026-11-17)

- Pin the `mcp` extra to `mcp>=1.2,<2`. mcp 2.x renamed FastMCP, so `saef-mcp` failed to start on a fresh install of 3.3.0.

## 3.3.0

- Added an MCP server, `saef-mcp` (module `saef.mcp_server`), so AI agents in Claude Desktop, Claude Code, Cursor and other MCP clients can estimate carbon, energy and water. Tools are `estimate_footprint`, `estimate_chat_usage`, `model_class`, `list_regions` and `compare_regions`, plus the resource `saef://about`.
- New optional extra `mcp` (`pip install "saef[mcp]"`). The core package is still dependency-free and imports `mcp` only inside `saef.mcp_server`.
- Dataset unchanged, still v3.1.0.

## 3.2.0

- Added `saef.integrations.langchain.SaefMiddleware` for LangChain v1 agents. It hooks `wrap_model_call` and `awrap_model_call`, records one estimate per model call, exposes `totals()` and `reset()`, and with `attach=True` writes the estimate to `AIMessage.response_metadata["saef"]`. The existing `SaefCallbackHandler` is unchanged.
- The `langchain` extra now installs `langchain>=1.0` (which includes `langchain-core`).
- Added `examples/quickstart.ipynb` and `examples/kaggle_notebook.ipynb`, executed notebooks that estimate a call, compare regions, use the LiteLLM callback and show how to cite.
- Dataset unchanged, still v3.1.0.

## 3.1.0

- Added LiteLLM, LangChain callback and response-object integrations.
- Dataset v3.1.0, a corrective release of v3.0.0 (Quebec regions mapped to the Quebec grid, zero-grid fallback for the Central African Republic).
