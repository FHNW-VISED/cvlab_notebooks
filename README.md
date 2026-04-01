# cvlab_notebooks

## Environment Setup

Install `uv` if not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### CPU (macOS)

Uses `requirements_no_gpu.txt` — GPU/CUDA and platform-incompatible packages are excluded.

```bash
uv venv .venv --python 3.12.12
uv pip install --python .venv/bin/python --requirements requirements_no_gpu.txt
source .venv/bin/activate
```

### GPU (Linux + NVIDIA)

Uses the full `requirements.txt` (generated from `pip list` on a Colab runtime with CUDA 12).

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --requirements requirements.txt
source .venv/bin/activate
```