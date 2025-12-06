---
tags:
- unsloth
- mlx
base_model: nightmedia/unsloth-Qwen3-VL-4B-Instruct-qx86x-hi-mlx
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: mlx
---


# aidana-slm-mlx

This is Qwen3-VL-4B-Instruct finetuned by Unsloth, with fixed chat template, qx86x-hi-mlx 8-bit quantized by nightmedia and further quantized to 4-bit with group size 32 by me.

The Deckard(qx) stores and most attention paths in low precision(6 bit), enhancing vital attention paths, head, context, and embeddings to 8 bit, and quantized with high precision(group size 32). With my quants embeddings are reduced to 4-bit and attention paths roughly 3 bit.

We're left with roughly 2.5 GB of model size (weights). With a small context, you're ending up with < 3 GB VRAM usage and about 37 to 40 tps on a Macbook Air M4 (base) while quality is mostly maintained. The model is able to hold simple conversation, solve math equations, understand images, and follow simple instructions (such as tool use and JSON schema-only output).

I've implemented a simple OSS inference server and WebUI for MLX-based language models: [kyr0/mlx-webui](https://github.com/kyr0/mlx-webui.git)

## Direct use with MLX

```bash
brew install uv
uv venv && source .venv/bin/activate
uv pip install mlx-lm
```

```python
# infer.py
from mlx_lm import load, generate

model, tokenizer = load("kyr0/aidana-slm-mlx")

prompt = "Hallo! Wie geht es dir?"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)
```

