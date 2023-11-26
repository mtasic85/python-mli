# CHANGELOG

## v0.1.10

Fixed:
    - Server: disabled `traceback` module usage.

## v0.1.9

Changed:
    - Client: Allow to access endpoints without checking for SSL, `verify_ssl=False`.

## v0.1.8

Changed:
    - Server: ENDPOINT needs to end with suffix like `/api/1.0`.

Fixed:
    - Server: disabled `traceback` module usage.

## v0.1.7

Added:
    - Server: `LlamaCppParams` support for `chatml` parameter.
    - Server: `CandleParams` support for `cpu` parameter.

Changed:
    - Example: `sync_demo.py` now uses kwargs instead of unpacking dict.
    - Server: Raise Error in case model does not exist.
    - mli: `params.py` types in separate module.

Fixed:
    - Server: Fixed memory leak on WebSocket routes.
    - Server: Wrong first characters of prompt output.
    - Client: IPv4 with PORT gets `http://` prefix.

Security:
    - Removed auto-download using `hf_hub_download` models because of securty risks.

## v0.1.6

Added:
    - Print `stderr` to debug output of ML engines.

Changed:
    - Examples: sync_demo.py, try `quantized` and `use_flash_attn`.

## v0.1.5

Added:
    - Server: `CandleParams` support now `quantized` and `use_flash_attn`.
    - Examples: Default ENDPOINT to `http://127.0.0.1:5000`.

## v0.1.4

Changed:
    - Client: auto-prefix (http / https) for BaseMLIClient's argument `endpoint`.

Fixed:
    - Removing echo-ed prompt to stdout for both `candle` and `llama.cpp`.

## v0.1.3

Added:
    - Server: download `GGUF` model if it does not exist using `huggingface-hub`.
    - Examples: `candle`-based sync examples `codellama`, `llama`, `mistral`.

Changed:
    - Instructions how to pre-download `GGUF` models required for `llama.cpp`.

Deprecated:
    - Server: `GGUF` model path is not passed anymore.

Fixed:
    - Server: engine `candle`, kind `llama`.
    - Server: engine `candle`, kind `mistral`.

## v0.1.2

Added:
    - Examples: sync_demo uses candle/llama.cpp for llama2/mistral/stablelm

Changed:
    - Server does not assert anymore model names/ids.

## v0.1.1

Added:
    - Install instructions for Debian, Ubuntu, ArchLinux, Manjaro, macOS.
    - Server: LlamaCppParams, CandleParams, LLMParams.
    - Examples: sync_demo.py, async_demo.py, langchain_sync_demo.py, langchain_async_demo.py

Changed:
    - Client examples.
    - Server: data/msg of type LLMParams.
    - Examples use context size 512.
    - Insomnia examples use context size 1024.
    - Package `mlipy` renamed to `mli`.
    - Streaming sync and async langchain functions.
    - Less noisy output on token generation.

## v0.1.0

Added:
    - Park first version.