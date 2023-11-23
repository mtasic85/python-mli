# python-mli

Python-based Machine Learning Interface


# Prerequisites

## Debian/Ubuntu
```bash
sudo apt update -y
sudo apt install build-essential rustc cargo git libssl-dev libffi-dev
```

1) Using internal repository:
```bash
sudo apt install python3.11 python3.11-dev python3.11-venv
```

2) Using external repository:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install python3.11 python3.11-dev python3.11-venv
```


## Arch/Manjaro

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
