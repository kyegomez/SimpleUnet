import torch
from unet.model import Unet

model = Unet(n_channels=3, n_classes=1)

inputs = torch.randn(1, 3, 256, 256)

outputs = model(inputs)

print(outputs.size)

print(outputs)