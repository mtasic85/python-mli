# mlipy

<!--
[![Build][build-image]]()
[![Status][status-image]][pypi-project-url]
[![Stable Version][stable-ver-image]][pypi-project-url]
[![Coverage][coverage-image]]()
[![Python][python-ver-image]][pypi-project-url]
[![License][mit-image]][mit-url]
-->
[![Downloads](https://img.shields.io/pypi/dm/mlipy)](https://pypistats.org/packages/mlipy)
[![Supported Versions](https://img.shields.io/pypi/pyversions/mlipy)](https://pypi.org/project/mlipy)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pure **Python**-based **Machine Learning Interface** for multiple engines with multi-modal support.

<!--
Python HTTP Server/Client (including WebSocket streaming support) for:
- [candle](https://github.com/huggingface/candle)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [LangChain](https://python.langchain.com)
-->

Python HTTP Server/Client (including WebSocket streaming support) for:
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [LangChain](https://python.langchain.com)


# Prerequisites

## Debian/Ubuntu

```bash
sudo apt update -y
sudo apt install build-essential git curl libssl-dev libffi-dev pkg-config
```

<!--
### Rust

1) Using latest system repository:

```bash
sudo apt install rustc cargo
```

2) Install rustup using official instructions:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup default stable
```
-->

### Python

1) Install Python using internal repository:
```bash
sudo apt install python3.11 python3.11-dev python3.11-venv
```

2) Install Python using external repository:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install python3.11 python3.11-dev python3.11-venv
```


<!--
## Arch/Manjaro

### Rust

1) Using latest system-wide rust/cargo:
```bash
sudo pacman -Sy base-devel openssl libffi git rust cargo rust-wasm wasm-bindgen
```

2) Using latest rustup:
```bash
sudo pacman -Sy base-devel openssl libffi git rustup
rustup default stable
```


## macOS


```bash
brew update
brew install rustup
rustup default stable
```
-->

# llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j
```


<!--
# candle

```bash
cd ~
git clone https://github.com/huggingface/candle.git
cd candle
find candle-examples/examples/llama/main.rs -type f -exec sed -i 's/print!("{prompt}")/eprint!("{prompt}")/g' {} +
find candle-examples/examples/phi/main.rs -type f -exec sed -i 's/print!("{prompt}")/eprint!("{prompt}")/g' {} +
find candle-examples/examples/mistral/main.rs -type f -exec sed -i -E 's/print\\!\\("\\{t\\}"\\)$/eprint\\!\\("\\{t\\}"\\)/g' {} +
find candle-examples/examples/stable-lm/main.rs -type f -exec sed -i -E 's/print\\!\\("\\{t\\}"\\)$/eprint\\!\\("\\{t\\}"\\)/g' {} +
find candle-examples -type f -exec sed -i 's/println/eprintln/g' {} +
cargo clean
```

CPU:
```bash
cargo build -r --bins --examples
```

GPU / CUDA:
```bash
cargo build --features cuda -r --bins --examples
```
-->


# Run Development Server

Setup virtualenv and install requirements:

```bash
git clone https://github.com/mtasic85/mlipy.git
cd mlipy

python3.11 -m venv venv
source venv/bin/activate
pip install poetry
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
poetry install
```

Download one of popular models to try them:

```bash
# NOTE: login in case you need to accept terms and conditions for some models
# huggingface-cli login

# command-r
huggingface-cli download mradermacher/c4ai-command-r-plus-i1-GGUF c4ai-command-r-plus.i1-IQ1_S.gguf
huggingface-cli download nold/c4ai-command-r-v01-GGUF c4ai-command-r-v01_Q3_K_M.gguf
huggingface-cli download nold/c4ai-command-r-v01-GGUF c4ai-command-r-v01_Q2_K.gguf

# xverse
huggingface-cli download xverse/XVERSE-7B-Chat-GGUF xverse-7b-chat-q4_k_m.gguf
huggingface-cli download xverse/XVERSE-13B-Chat-GGUF xverse-13b-chat-q4_k_m.gguf

# internlm2
huggingface-cli download nold/internlm2-chat-20b-GGUF internlm2-chat-20b_Q3_K_M.gguf
huggingface-cli download nold/internlm2-chat-20b-GGUF internlm2-chat-20b_Q4_K_M.gguf
huggingface-cli download izumi04/InternLM2-Chat-7B-GGUF internlm2-chat-7b-Q3_K_M.gguf
huggingface-cli download izumi04/InternLM2-Chat-7B-GGUF internlm2-chat-7b-Q4_K_M.gguf

# yi
huggingface-cli download LoneStriker/Yi-9B-200K-GGUF Yi-9B-200K-Q4_K_M.gguf
huggingface-cli download LoneStriker/Yi-6B-200K-GGUF Yi-6B-200K-Q4_K_M.gguf

# gemma
huggingface-cli download pabloce/dolphin-2.8-gemma-2b-GGUF dolphin-2.8-gemma-2b.Q4_K_M.gguf
huggingface-cli download bartowski/gemma-1.1-7b-it-GGUF gemma-1.1-7b-it-Q4_K_M.gguf
huggingface-cli download bartowski/gemma-1.1-2b-it-GGUF gemma-1.1-2b-it-Q4_K_M.gguf

# qwen
huggingface-cli download qwp4w3hyb/Qwen1.5-14B-Chat-iMat-GGUF qwen1.5-14b-chat-imat-IQ1_S.gguf
huggingface-cli download qwp4w3hyb/Qwen1.5-14B-Chat-iMat-GGUF qwen1.5-14b-chat-imat-IQ2_XS.gguf
huggingface-cli download qwp4w3hyb/Qwen1.5-14B-Chat-iMat-GGUF qwen1.5-14b-chat-imat-IQ2_S.gguf
huggingface-cli download qwp4w3hyb/Qwen1.5-14B-Chat-iMat-GGUF qwen1.5-14b-chat-imat-IQ2_M.gguf
huggingface-cli download qwp4w3hyb/Qwen1.5-14B-Chat-iMat-GGUF qwen1.5-14b-chat-imat-IQ3_M.gguf
huggingface-cli download Qwen/Qwen1.5-14B-Chat-GGUF qwen1_5-14b-chat-q2_k.gguf
huggingface-cli download Qwen/Qwen1.5-14B-Chat-GGUF qwen1_5-14b-chat-q3_k_m.gguf
huggingface-cli download Qwen/Qwen1.5-14B-Chat-GGUF qwen1_5-14b-chat-q4_k_m.gguf
huggingface-cli download Qwen/Qwen1.5-7B-Chat-GGUF qwen1_5-7b-chat-q4_k_m.gguf
huggingface-cli download Qwen/Qwen1.5-4B-Chat-GGUF qwen1_5-4b-chat-q4_k_m.gguf
huggingface-cli download Qwen/Qwen1.5-1.8B-Chat-GGUF qwen1_5-1_8b-chat-q4_k_m.gguf
huggingface-cli download Qwen/Qwen1.5-0.5B-Chat-GGUF qwen1_5-0_5b-chat-q4_k_m.gguf

# mistral ai
huggingface-cli download bartowski/Mistral-22B-v0.2-GGUF Mistral-22B-v0.2-IQ2_M.gguf
huggingface-cli download bartowski/Mistral-22B-v0.2-GGUF Mistral-22B-v0.2-Q4_K_M.gguf
huggingface-cli download TheBloke/dolphin-2.7-mixtral-8x7b-GGUF dolphin-2.7-mixtral-8x7b.Q3_K_M.gguf
huggingface-cli download mradermacher/Mixtral-8x7B-Instruct-v0.1-i1-GGUF Mixtral-8x7B-Instruct-v0.1.i1-IQ1_S.gguf
huggingface-cli download mradermacher/Mixtral-8x7B-Instruct-v0.1-i1-GGUF Mixtral-8x7B-Instruct-v0.1.i1-IQ2_XXS.gguf
huggingface-cli download mradermacher/Mixtral-8x7B-Instruct-v0.1-i1-GGUF Mixtral-8x7B-Instruct-v0.1.i1-IQ2_M.gguf
huggingface-cli download mradermacher/Mixtral-8x7B-Instruct-v0.1-i1-GGUF Mixtral-8x7B-Instruct-v0.1.i1-Q3_K_M.gguf
huggingface-cli download bartowski/dolphin-2.8-mistral-7b-v02-GGUF dolphin-2.8-mistral-7b-v02-Q4_K_M.gguf
huggingface-cli download TheBloke/dolphin-2.6-mistral-7B-GGUF dolphin-2.6-mistral-7b.Q4_K_M.gguf
huggingface-cli download NousResearch/Hermes-2-Pro-Mistral-7B-GGUF Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf

# stability ai
huggingface-cli download lmz/candle-stablelm
huggingface-cli download stabilityai/stablelm-2-12b-chat-GGUF stablelm-2-12b-chat-Q4_K_M.gguf
huggingface-cli download brittlewis12/stablelm-2-1_6b-chat-GGUF stablelm-2-1_6b-chat.Q8_0.gguf
huggingface-cli download stabilityai/stablelm-2-zephyr-1_6b stablelm-2-zephyr-1_6b-Q4_1.gguf
huggingface-cli download stabilityai/stablelm-2-zephyr-1_6b stablelm-2-zephyr-1_6b-Q8_0.gguf
huggingface-cli download TheBloke/stablelm-zephyr-3b-GGUF stablelm-zephyr-3b.Q4_K_M.gguf
huggingface-cli download TheBloke/stable-code-3b-GGUF stable-code-3b.Q4_K_M.gguf

# technology innovation institute (tii)
huggingface-cli download mradermacher/falcon-40b-instruct-GGUF falcon-40b-instruct.IQ3_XS.gguf
huggingface-cli download maddes8cht/tiiuae-falcon-7b-instruct-gguf tiiuae-falcon-7b-instruct-Q4_K_M.gguf

# meta llama
huggingface-cli download NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
huggingface-cli download bartowski/dolphin-2.9-llama3-8b-1m-GGUF dolphin-2.9-llama3-8b-1m-IQ1_S.gguf
huggingface-cli download PrunaAI/dolphin-2.9-llama3-8b-1m-GGUF-smashed dolphin-2.9-llama3-8b-1m.Q4_K_M.gguf
huggingface-cli download PrunaAI/dolphin-2.9-llama3-8b-256k-GGUF-smashed dolphin-2.9-llama3-8b-256k.IQ3_XS.gguf
huggingface-cli download PrunaAI/dolphin-2.9-llama3-8b-256k-GGUF-smashed dolphin-2.9-llama3-8b-256k.Q4_K_M.gguf
huggingface-cli download mradermacher/Meta-Llama-3-8B-Instruct-64k-GGUF Meta-Llama-3-8B-Instruct-64k.Q4_K_M.gguf
huggingface-cli download cognitivecomputations/dolphin-2.9-llama3-8b-gguf dolphin-2.9-llama3-8b-q4_K_M.gguf
huggingface-cli download cognitivecomputations/dolphin-2.9-llama3-8b-gguf dolphin-2.9-llama3-8b-q8_0.gguf
huggingface-cli download mradermacher/Meta-Llama-3-8B-Instruct-i1-GGUF Meta-Llama-3-8B-Instruct.i1-Q4_K_M.gguf
huggingface-cli download mradermacher/Meta-Llama-3-8B-Instruct-i1-GGUF Meta-Llama-3-8B-Instruct.i1-IQ4_XS.gguf
huggingface-cli download TheBloke/Orca-2-7B-GGUF orca-2-7b.Q4_K_M.gguf
huggingface-cli download afrideva/MiniChat-2-3B-GGUF minichat-2-3b.q4_k_m.gguf
huggingface-cli download azarovalex/MobileLLaMA-1.4B-Chat-GGUF MobileLLaMA-1.4B-Chat-Q4_K.gguf
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q8_0.gguf
huggingface-cli download s3nh/TinyDolphin-2.8-1.1b-GGUF tinydolphin-2.8-1.1b.Q4_K_M.gguf
huggingface-cli download s3nh/TinyDolphin-2.8-1.1b-GGUF tinydolphin-2.8-1.1b.Q8_0.gguf
huggingface-cli download thephimart/tinyllama-4x1.1b-moe.Q5_K_M.gguf tinyllama-4x1.1b-moe.Q5_K_M.gguf

# microsoft phi
huggingface-cli download lmz/candle-quantized-phi
huggingface-cli download PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed Phi-3-mini-128k-instruct.IQ2_XXS.gguf
huggingface-cli download PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed Phi-3-mini-128k-instruct.Q4_K_M.gguf
huggingface-cli download PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed Phi-3-mini-128k-instruct.Q5_K_M.gguf
huggingface-cli download QuantFactory/Phi-3-mini-128k-instruct-GGUF Phi-3-mini-128k-instruct.Q4_K_M.gguf
huggingface-cli download QuantFactory/Phi-3-mini-128k-instruct-GGUF Phi-3-mini-128k-instruct.Q8_0.gguf
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-fp16.gguf
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf
huggingface-cli download TheBloke/dolphin-2_6-phi-2-GGUF dolphin-2_6-phi-2.Q4_K_M.gguf
huggingface-cli download MaziyarPanahi/phi-2-super-GGUF phi-2-super.Q4_K_M.gguf
huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf
huggingface-cli download TKDKid1000/phi-1_5-GGUF phi-1_5-Q4_K_M.gguf
```

Run server:

```bash
python -B -m mli.server --llama-cpp-path='~/llama.cpp'
```


# Run Examples

Using GPU:

```bash
NGL=99 python -B examples/sync_demo.py
```

Using CPU:

```bash
python -B examples/sync_demo.py
python -B examples/async_demo.py
python -B examples/langchain_sync_demo.py
python -B examples/langchain_async_demo.py
```


# Run Production Server

## Generate self-signed SSL certificates

```bash
openssl req -x509 -nodes -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
```



## Run

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U mlipy
python -B -m mli.server
```
