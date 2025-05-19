from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
distilgpt2_path = "app/distilgpt2"
onnx_model_path = "app/distilgpt2.onnx"
quantized_model_path = "app/distilgpt2_quantized.onnx"

# Load pretrained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(distilgpt2_path)
tokenizer = AutoTokenizer.from_pretrained(distilgpt2_path)
model.eval()

# Dummy input for tracing
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# Export to ONNX
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    (inputs["input_ids"],),
    onnx_model_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                  "logits": {0: "batch_size", 1: "sequence_length"}},
    do_constant_folding=True,
    opset_version=13
)
print(f"ONNX model saved to {onnx_model_path}")

# Apply dynamic quantization to ONNX model
print("Applying ONNX quantization...")
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8  # use QInt8 for best compression
)
print(f"Quantized ONNX model saved to {quantized_model_path}")

# Compare sizes
original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)
print(f"Original ONNX model size: {original_size:.2f} MB")
print(f"Quantized ONNX model size: {quantized_size:.2f} MB")

# Save tokenizer (needed for inference)
tokenizer.save_pretrained("app/distilgpt2_tokenizer")
print("Tokenizer saved.")
