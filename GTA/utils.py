import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Grayscale

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--baseline', action='store_true', default=False, help='whether to run baseline')
    
    parser.add_argument('--width', type=int, default=32, help='width of input images')
    parser.add_argument('--channels', type=int, default=3, choices=[1, 3], help='number of channels in input images')
    parser.add_argument('--filters', type=int, default=64, help='number of filters in encoder network')
    
    parser.add_argument('--embedding_dim', type=int, default=128, help='size of encoder network output vector')
    parser.add_argument('--latent_dim', type=int, default=512, help='size of generator network input vector')

    parser.add_argument('--lr', type=float, default=0.0005, help='adam optimizer learning rate')
    parser.add_argument('--beta1', type=float, default=0.8, help='adam optimizer beta1 value')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam optimizer beta2 value')
    parser.add_argument('--batch_size', type=int, default=100, help='training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    
    parser.add_argument('--adv_weight', type=float, default=0.1, help='weight of source adversarial loss')
    parser.add_argument('--alpha', type=float, default=0.3, help='multiplicative weight of target adversarial loss')
    
    parser.add_argument('--device', type=str, default='cuda', help='specify the device for pytorch tensors')
    return parser.parse_args()


def init_weights(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        size = layer.weight.size()
        layer.weight.data.normal_(0.0, 0.1)
        layer.bias.data.fill_(0)
        
def get_transform(width, channels):
    return Compose([Resize(width), 
                    Grayscale(channels),
                    ToTensor(), 
                    Normalize([0.5]*channels, [0.5]*channels)]
    )