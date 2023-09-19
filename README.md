[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Unet
My implemenetation of a modular Unet from the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"

[Paper Link](https://arxiv.org/abs/1505.04597)

# Appreciation
* Lucidrains
* Agorians


# Install
`pip install unet-torch`

# Usage
```python
import torch
from unet.model import Unet

model = Unet(n_channels=3, n_classes=1)

inputs = torch.randn(1, 3, 256, 256)

outputs = model(inputs)

print(outputs.size)

print(outputs)

```

# License
MIT

# Citations
```bibtex
@misc{1505.04597,
Author = {Olaf Ronneberger and Philipp Fischer and Thomas Brox},
Title = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
Year = {2015},
Eprint = {arXiv:1505.04597},
}
```