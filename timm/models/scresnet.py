import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .factory import create_model
from .registry import register_model

# __all__ = ['ScResnet']  # model_registry will add each entrypoint fn to this

default_cfg = {
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
    'crop_pct': 0.875, 'interpolation': 'bilinear',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
}

def gumbel_softmax(x):
    # TODO: Implement Gumbel-Softmax
    pass

def saliency():
    pass

class ScResnet(nn.Module):
    def __init__(self, classifier='resnet18', num_classes=1000, in_chans=3) -> None:
        super().__init__()
        self.resnet = create_model(classifier)

    def forward(self, x):
        x = self.resnet(x)
        return x

@register_model
def scresnet(pretrained=False, **kwargs):
    return ScResnet()