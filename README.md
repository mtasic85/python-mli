# python-mli

Python-based Machine Learning Interface

# Prerequisites

Debian/Ubuntu:
```bash
sudo apt update -y
apt install build-essential rustc cargo git
```

```bash
sudo pacman -Sy base-devel rust cargo rust-wasm wasm-bindgen
```

# llama.cpp

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

# candle

```bash
cd ~
git clone https://github.com/huggingface/candle.git
cd candle
cargo build -r --bins --examples
```

# Run mlipy Server

```bash
pip install -U mlipy
```
