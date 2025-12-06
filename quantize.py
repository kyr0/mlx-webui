from mlx_lm import load
from mlx_lm.utils import dequantize_model, quantize_model, save
import shutil
import os

model_path = "nightmedia/unsloth-Qwen3-VL-4B-Instruct-qx86x-hi-mlx"
mlx_path = "./quantized_model"

# Clean up previous attempt
if os.path.exists(mlx_path):
    shutil.rmtree(mlx_path)

print("Loading model...")
model, tokenizer, config = load(model_path, return_config=True)

print("Dequantizing model...")
model = dequantize_model(model)
# Remove old quantization config
config.pop("quantization", None)
config.pop("quantization_config", None)

print("Quantizing model to 4 bits...")
# 4-bit quantization
q_group_size = 32 # keep deckard
q_bits = 4

model, config = quantize_model(
    model, 
    config, 
    group_size=q_group_size, 
    bits=q_bits
)

print("Saving model...")
save(
    mlx_path,
    src_path_or_repo=model_path,
    model=model,
    tokenizer=tokenizer,
    config=config
)
print("Done.")