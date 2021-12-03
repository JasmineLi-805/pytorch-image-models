import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .factory import create_model
from .registry import register_model
from .helpers import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# __all__ = ['ScResnet']  # model_registry will add each entrypoint fn to this

default_cfg = {
    'num_classes': 1000,
    'input_size': (4, 32, 32), 
    'crop_pct': 1.0, 'interpolation': 'bicubic',
    # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
    'mean': (0, 0, 0, 0),
    'std': (1, 1, 1, 1)
}
model_cfg = {
    'SC_layers': [
        # in_chan, out_chan, kernel_size, stride, activation
        [ 1,  8, 3, 2, 'relu6'],
        [ 8, 16, 3, 2, 'relu6'],
        [16, 32, 3, 2, 'relu6']
    ],
    'downsample_size': (1, 32, 32),
    'original_size': (3, 32, 32),
    'num_classes': default_cfg['num_classes'],
    'classifier': 'resnet18'
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
        super().__init__()
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
                act = nn.ReLU6()
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
        # TODO: switch to Gumbel softmax
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x: (batch, crop_n, channel, height, width)
        batch_size = x.shape[0]
        n = x.shape[1]
        x = x.view(batch_size * n, x.shape[2], x.shape[3], x.shape[4])
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(batch_size, n)
        # TODO: apply gumbel softmax
        x = self.softmax(x)
        return x

class ScResnet(nn.Module):
    def __init__(self, original_size, downsample_size, SC_layers,
                 classifier='resnet18', num_classes=1000, in_chans=4) -> None:
        super().__init__()
        self.down_size = downsample_size
        self.orig_size = original_size
        self.num_classes=num_classes

        self.saliency_map = ScLayer(SC_layers, self.down_size)
        self.resnet = create_model(classifier)

    def forward(self, x):
        print(f'the shape of input img={x.shape}')
        x = self.resnet(x)
        return x

@register_model
def scresnet(**kwargs):
    pretrained = False
    return build_model_with_cfg(ScResnet, 'scresnet', pretrained, default_cfg, **model_cfg)