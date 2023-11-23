# python-mli

Python-based Machine Learning Interface


# Prerequisites

## Debian/Ubuntu

```bash
sudo apt update -y
sudo apt install build-essential git curl libssl-dev libffi-dev pkg-config
```

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


# Run Development Server

```bash
git clone https://github.com/mtasic85/mlipy.git
cd mlipy

python3.11 -m venv venv
source venv/bin/activate
pip install poetry
poetry install
python -B mlipy/server.py
```


# Run mlipy Server

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U mlipy
python -m mlipy.server
```
