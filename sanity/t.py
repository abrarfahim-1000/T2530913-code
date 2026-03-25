import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
elif torch.xpu.is_available():
    print(f"XPU available: {torch.xpu.is_available()}")