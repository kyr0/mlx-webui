from mlx_lm import load
import json

model_path = "nightmedia/unsloth-Qwen3-VL-4B-Instruct-qx86x-hi-mlx"
model, tokenizer, config = load(model_path, return_config=True)

print("Config quantization:", config.get("quantization"))
print("First layer type:", type(model.layers[0]))
if hasattr(model.layers[0], "self_attn"):
    print("Self attention q_proj type:", type(model.layers[0].self_attn.q_proj))
