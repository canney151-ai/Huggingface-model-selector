# Huggingface Model Selector

A lightweight Flask web app for browsing and downloading GGUF models from HuggingFace directly to your local machine.

## Features

- Search HuggingFace for GGUF-compatible models
- Browse available quantizations per model with file sizes
- Recommended quantizations highlighted (Q4_K_M, Q5_K_M, IQ4_XS, etc.)
- Real-time download progress with speed indicator
- Skips files too large to fit in available memory
- Manage downloaded models (view, delete)
- Optional HuggingFace token support for gated models

## Requirements

- Python 3.10+
- Flask 3.0+
- requests 2.31+

## Setup

```bash
git clone https://github.com/canney151-ai/Huggingface-model-selector.git
cd Huggingface-model-selector
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## Configuration

Settings are stored at `~/.config/hf-gguf-dl/config.json`.

| Setting | Default | Description |
|---|---|---|
| `models_dir` | `~/models` | Where downloaded models are saved |
| `hf_token` | _(empty)_ | HuggingFace API token for gated models |

Both can be configured from the Settings page in the UI.

## Hardware

Designed for machines with large unified memory (tested on AMD Strix Halo with 96 GB). Files larger than 88 GB are automatically filtered out.
