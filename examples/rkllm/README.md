# RKLLM Workflow (Conversion + C++/Python Inference on Axon)

This `rkllm/` folder contains:
- `convert.py`: converts a Hugging Face model directory to `.rkllm`
- `inference.cpp`: RKLLM C++ runtime inference app
- `inference.py`: RKLLM Python runtime inference app (ctypes wrapper)
- `dataset.json`: optional calibration dataset
- `rkllm.h` + `librkllmrt.so`: RKLLM C++/Python runtime build dependencies

Recommended flow:
1. Convert once on host machine
2. Copy the generated `.rkllm` model to Axon
3. Run inference on Axon using either C++ or Python

## 0) Get Started

```bash
git clone https://github.com/vicharak-in/Axon-NPU-Guide.git
cd Axon-NPU-Guide/rkllm
```

`rkllm/` is the working folder for this guide and includes the required runtime files.

## 1) Common Conversion

### 1.1 Create environment + get toolkit

```bash
python3 -m venv venv-rkllm
source venv-rkllm/bin/activate

git clone https://github.com/airockchip/rknn-llm.git
```

If using Python 3.12:

```bash
export BUILD_CUDA_EXT=0
pip install rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl
```

If you hit `No module named pkg_resources`:

```bash
pip install "setuptools==68.0.0"
```

### 1.2 Download model from Hugging Face

```bash
sudo apt install -y git-lfs
git lfs install

git clone https://huggingface.co/Qwen/Qwen3-0.6B
# Example alternative:
# git clone https://huggingface.co/Qwen/Qwen2-1.5B
```

### 1.3 Convert to RKLLM

Qwen3-0.6B example:

```bash
python3 convert.py -i ./Qwen3-0.6B -o <output-file-name.rkllm> --device cpu --dtype float32 --quantized-dtype w8a8 --quantized-algorithm normal --optimization-level 1 --num-npu-core 3 --target-platform rk3588 --max-context 4096
```
Add the flag: --dataset <path/to/dataset.json> when using a calibration dataset.

Notes:
- Use `--dataset dataset.json` to enable calibration dataset quantization.
- `--max-context` must be `>0`, `<=16384`, and a multiple of `32`.
- `--quantized-algorithm grq/gdq` requires `--device cuda` in `convert.py`.

After conversion, copy only the generated `.rkllm` model file to your Axon `rkllm/` folder.

---

## 2) C++ Inference on Axon

### 2.1 Compile

```bash
g++ -O2 -std=c++17 -I. inference.cpp -L. -lrkllmrt -Wl,-rpath,'$ORIGIN' -o inference
```
> Note: keep the `librkllmrt.so` file and the `rkllm.h` file in the same directory as the inference.cpp file for the above command to work.

### 2.2 Run

```bash
./inference --model <path/to/model.rkllm> --target-platform rk3588 --stream --print-perf --keep-history
```

Useful behavior:
- If `--prompt` is not passed, it starts interactive mode.
- Interactive commands:
  - `exit`
  - `clear` (clears KV cache)

---

## 3) Python Inference on Axon (with venv)

`inference.py` uses only Python stdlib + `librkllmrt.so`, so a lightweight venv is enough.

### 3.1 Create env

```bash
python3 -m venv venv-rkllm
source venv-rkllm/bin/activate
```

### 3.2 Inference

Single-shot prompt

```bash
python3 inference.py -m <path/to/model.rkllm> --target-platform rk3588 --stream --print-perf --prompt "Hello"
```

Keep chat history across turns (interactive mode):

```bash
python3 inference.py -m <path/to/model.rkllm> --target-platform rk3588 --stream --print-perf --keep-history
```

---

## 4) Troubleshooting

- `ModuleNotFoundError: No module named 'pkg_resources'`
  - Run: `pip install "setuptools==68.0.0"`

- `OSError: librkllmrt.so: cannot open shared object file`
  - Pass `--runtime-lib /full/path/librkllmrt.so` or set `LD_LIBRARY_PATH`.
  - Confirm you are using the Linux `aarch64` runtime on Axon.
---