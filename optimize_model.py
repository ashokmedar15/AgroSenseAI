import torch
import torch.nn as nn
from torchvision import models
import torch.quantization
import os

# Load the original model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
expected_num_classes = 27  # Update based on your class count

# Replace the final classifier layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, expected_num_classes)
)

# Load trained weights
model_path = r"app/multi_class_classifier2.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to eval mode before quantization

# ⚠️ Only dynamically quantize Linear layers (Conv2d not supported)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
print("Dynamic quantization completed.")

# Convert to TorchScript
scripted_model = torch.jit.script(quantized_model)
optimized_model_path = r"app/multi_class_classifier2_optimized.pth"
scripted_model.save(optimized_model_path)
print(f"Optimized model saved to {optimized_model_path}")

# Compare model file sizes
original_size = os.path.getsize(model_path) / (1024 * 1024)
optimized_size = os.path.getsize(optimized_model_path) / (1024 * 1024)
print(f"Original model size: {original_size:.2f} MB")
print(f"Optimized model size: {optimized_size:.2f} MB")
