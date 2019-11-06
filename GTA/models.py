import torch
import torch.nn as nn
import torch.optim as optim
from .layers import ACModule
from .utils import init_weights

def get_generator(latent_dim, embedding_dim, filters, nclasses, lr, betas, device):
    G = nn.Sequential(
        nn.ConvTranspose2d(latent_dim + embedding_dim + nclasses+1, filters*8, 2, 1, 0, bias=False),
        nn.BatchNorm2d(filters*8),
        nn.ReLU(True),
    
        nn.ConvTranspose2d(filters*8, filters*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(filters*4),
        nn.ReLU(True),
    
        nn.ConvTranspose2d(filters*4, filters*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(filters*2),
        nn.ReLU(True),
    
        nn.ConvTranspose2d(filters*2, filters, 4, 2, 1, bias=False),
        nn.BatchNorm2d(filters),
        nn.ReLU(True),
    
        nn.ConvTranspose2d(filters, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    ).apply(init_weights).to(device)
    
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=betas)
    
    return G, G_optimizer


def get_discriminator(filters, nclasses, lr, betas, device):
    D = nn.Sequential(
        nn.Conv2d(3, filters, 3, 1, 1),            
        nn.BatchNorm2d(filters),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2,2),
    
        nn.Conv2d(filters, filters*2, 3, 1, 1),         
        nn.BatchNorm2d(filters*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2,2),
        
    
        nn.Conv2d(filters*2, filters*4, 3, 1, 1),           
        nn.BatchNorm2d(filters*4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(filters*4, filters*2, 3, 1, 1),           
        nn.BatchNorm2d(filters*2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(4,4),
        ACModule(filters, nclasses)
    ).apply(init_weights).to(device)
    
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=betas)
    
    return D, D_optimizer


def get_encoder(channels, filters, embedding_dim, lr, betas, device):
    F = nn.Sequential(
        nn.Conv2d(channels, filters, 5, 1, 0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(filters, filters, 5, 1, 0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(filters, embedding_dim, 5, 1,0),
        nn.ReLU(inplace=True)
    ).apply(init_weights).to(device)
    
    F_optimizer = optim.Adam(F.parameters(), lr=lr, betas=betas)
    
    return F, F_optimizer


def get_classifier(filters, nclasses, lr, betas, device):
    C = nn.Sequential(
        nn.Linear(2*filters, 2*filters),
        nn.ReLU(inplace=True),
        nn.Linear(2*filters, nclasses),      
    ).apply(init_weights).to(device)
    
    C_optimizer = optim.Adam(C.parameters(), lr=lr, betas=betas)
    
    return C, C_optimizer