# This is the implementation for L-AutoDA

## Required Package
Before running the code, please install the following packages:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm
pip install foolbox
```
`foolbox`is used for adversarial attack.

`timm` is used for loading pretrained models for later test.

`torch` is used for running neural networks.