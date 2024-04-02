# CHANGELOG

## v0.1.47

Changed:
    - Updated all dependencies. 
    - Removed `sentencepiece` as dependencies.

## v0.1.46

Fixed:
    - The eight approach to terminate `proc` subprocess.

## v0.1.45

Fixed:
    - The seventh approach to terminate `proc` subprocess.

## v0.1.44

Fixed:
    - The sixth approach to terminate `proc` subprocess.

## v0.1.43

Fixed:
    - The fifth approach to terminate `proc` subprocess.

## v0.1.42

Fixed:
    - The forth approach to terminate `proc` subprocess.

## v0.1.41

Fixed:
    - The third approach to terminate `proc` subprocess.

## v0.1.40

Fixed:
    - The second approach to terminate `proc` subprocess.

## v0.1.39

Fixed:
    - Fixed terminating `proc` subprocess.

## v0.1.38

Changed:
    - Renamed `LLMParams` to `ModelParams`.

## v0.1.37

Fixed:
    - `llama.cpp` prompt/messages and image when saved in file remove `prompt`, `messages` and `image` from params.

## v0.1.36

Fixed:
    - `llama.cpp` prompt/messages and image removed from temp file(s).

## v0.1.35

Added:
    - `llama.cpp` prompt/messages and image written in temp file(s).

## v0.1.34

Added:
    - `LlamaCppParams.file` Prompt go to temp prompt file because of CLI argument limit in terminal/shell.
    - `LlamaCppParams.image` Prompt image go to temp prompt file.
    - `LlamaCppParams.mmproj` Path to a multimodal projector file for LLaVA.

## v0.1.33

Fixed:
    - Docker environment without GPUs.

## v0.1.32

Changed:
    - Do not import `server.py` from package by default.

## v0.1.31

Added:
    - Detect `AMD` and `NVIDIA` GPUs because `llama.cpp` needs to know valid arguments.

## v0.1.30

Changed:
    - Default port is `4000` instead of `5000`.

## v0.1.29

Added:
    - `client` special `model_id='echo/echo'` which returns same text back without going to server.
    - examples/defs.py has new grammar templates.

Changed:
    - `server` uses `web.RouteTableDef` for defining routes.

## v0.1.28

Added:
    - `amazon/MistralLite` 7B model demos.
    - `google/gemma` 7B and 2B models demos.

Deprecated:
    - `candle` models in `README.md`.
    - `candle` instructions in `README.md`.

Fixed:
    - `llama.cpp`/`candle` engine does not handle wrong/missing model.
    - Early check if engine one of `llama.cpp` or `candle`.

## v0.1.27

Added:
    - Added `llama.cpp` parameters: `seed`, `threads`, `grammar`, `grammar_file`, `cfg_negative_prompt`, `cfg_scale`, `rope_scaling`, `rope_scale`, `rope_freq_base`, `rope_freq_scale`, `cont_batching`.

Changed:
    - `ctx_size` is not `0`, size of the prompt context (default: 512, 0 = loaded from model)

## v0.1.26

Added:
    - `pyproject.toml` initial support for `torch` (WIP).

Fixed:
    - `examples/sync_demo.py` now uses correct models for `llama.cpp`.

## v0.1.25

Changed:
    - `format_messages` using `creator_model_id`

Fixed:
    - Formatter will use `creator_model_id` if `model_id` is misssing.
    - New "simple" default/fallback formatting if `model_id`/`creator_model_id` is missing.

## v0.1.24

Fixed:
    - `executable` was not patched in `msg`.

## v0.1.23

Fixed:
    - Supported old `kind` paramater which can be used for `executable`.

## v0.1.22

Changed:
    - params: `kind` is not `executable`.
    - `langchain` package is optional now, client code is move to `langchain_client.py`.
    - `uvloop` package is optional now.

Fixed:
    - Formatting of chat messages (role-based) using `transformers.AutoTokenizer.apply_chat_template`.

## v0.1.21

Added:
    - server/client: `split_mode`, `tensor_split`, `main_gpu`.

Changed:
    - README: New models.

## v0.1.20

Changed:
    - README: instructions how to build `llama.cpp`
    - README: New models.

## v0.1.19

Changed:
    - README: instructions how to build `llama.cpp` and `candle`.
    - examples: more examples and improvements in `sync_demo.py`.
    - server: fixed format output for `messages_syntax='llama'`.

Fixed:
    - server: no need to parse printed/echoed prompt thanks to new instructions how to build `llama.cpp` and `candle`.

## v0.1.18

Changed:
    - server: improved `messages_syntax` formatting.

Fixed:
    - server: better handling of `stop` words.

## v0.1.17

Fixed:
    - server: optional `messages_syntax`.

## v0.1.16

Changed:
    - server: support for `messages_syntax` formating `chatml`, `llama`, `zephyr`.

Fixed:
    - client: fixed `llama.cpp` examples, `ctx_size`.

## v0.1.15

Fixed:
    - server: fixed finding position of printed initial prompt, so it can be skipped.

## v0.1.14

Fixed:
    - server: fixed trying to read `stderr` when `proc is None`.

## v0.1.13

Fixed:
    - server: fixed `stop` / `stop_enc` logic while streaming.

## v0.1.12

Fixed:
    - server: fixed `stop` logic while streaming.

## v0.1.11

Changed:
    - Server: enabled `traceback` module usage.

Fixed:
    - params: added missing `engine` field.

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