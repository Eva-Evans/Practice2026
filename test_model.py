import torch

from src.model.lcnn import LCNN

model = LCNN()
x = torch.randn(4, 1, 863, 600)
y = model(x)
print(f"Output shape: {y.shape}")
