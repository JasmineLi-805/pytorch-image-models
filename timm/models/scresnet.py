import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .factory import create_model
from .registry import register_model
from .helpers import build_model_with_cfg

# __all__ = ['ScResnet']  # model_registry will add each entrypoint fn to this

default_cfg = {
    'num_classes': 1000,
    'downsample_size': (1, 128, 128),
    'original_size': (3, 224, 224),
    'SC_layers': [
        # in_chan, out_chan, kernel_size, stride, activation
        [ 1,  8, 3, 2, 'relu6'],
        [ 8, 16, 3, 2, 'relu6'],
        [16, 32, 3, 2, 'relu6']
    ]
}

def gumbel_softmax(x):
    # TODO: Implement Gumbel-Softmax
    pass

def calc_conv_out_dim(in_dim, kernel, stride):
    out_dim = in_dim - (kernel - 1) - 1
    out_dim = 1.0 * out_dim / stride
    out_dim = math.floor(out_dim + 1)
    return out_dim

class ScLayer(nn.Module):
    def __init__(self, cfg, input_size):
        c, h, w = input_size
        block_list = []
        for layer_cfg in cfg:
            conv = nn.Conv2d(
                in_channels=layer_cfg[0], 
                out_channels=layer_cfg[1],
                kernel_size=layer_cfg[2],
                stride=layer_cfg[3]
            )
            bn = nn.BatchNorm2d(layer_cfg[1])
            act = None
            if layer_cfg[4] == 'relu6':
                act = nn.Relu6()
            # append the layers
            block_list.append(conv)
            block_list.append(bn)
            block_list.append(act)
            # calculate current image size
            c = layer_cfg[1]
            h = calc_conv_out_dim(h, layer_cfg[2], layer_cfg[3])
            w = calc_conv_out_dim(w, layer_cfg[2], layer_cfg[3])
        self.blocks = nn.Sequential(*block_list)
        self.fc = nn.Linear(c * h * w, 1)
    
    def forward(self, x):
        # x: (batch, channel, height, width)
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class ScResnet(nn.Module):
    def __init__(self, original_size, downsample_size, SC_layers,
                 classifier='resnet18', num_classes=1000) -> None:
        super().__init__()
        self.resnet = create_model(classifier)
        self.num_classes=num_classes

    def forward(self, x):
        x = self.resnet(x)
        return x

@register_model
def scresnet(**kwargs):
    pretrained = False
    return build_model_with_cfg(ScResnet, 'scresnet', pretrained, default_cfg, **kwargs)