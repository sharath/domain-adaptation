import os
from torchvision.datasets import MNIST, SVHN
os.makedirs('datasets', exist_ok=True)
SVHN(root='datasets/', split='train', download=True)
SVHN(root='datasets/', split='test', download=True)
MNIST(root='datasets/', train=True, download=True)