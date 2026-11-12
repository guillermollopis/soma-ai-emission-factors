# Changelog

## 3.2.0

- Added `saef.integrations.langchain.SaefMiddleware` for LangChain v1 agents. It hooks `wrap_model_call` and `awrap_model_call`, records one estimate per model call, exposes `totals()` and `reset()`, and with `attach=True` writes the estimate to `AIMessage.response_metadata["saef"]`. The existing `SaefCallbackHandler` is unchanged.
- The `langchain` extra now installs `langchain>=1.0` (which includes `langchain-core`).
- Added `examples/quickstart.ipynb` and `examples/kaggle_notebook.ipynb`, executed notebooks that estimate a call, compare regions, use the LiteLLM callback and show how to cite.
- Dataset unchanged, still v3.1.0.

## 3.1.0

- Added LiteLLM, LangChain callback and response-object integrations.
- Dataset v3.1.0, a corrective release of v3.0.0 (Quebec regions mapped to the Quebec grid, zero-grid fallback for the Central African Republic).
