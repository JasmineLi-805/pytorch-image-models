import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18

from .factory import create_model
from .registry import register_model

# __all__ = ['ScResnet']  # model_registry will add each entrypoint fn to this

def gumbel_softmax(x):
    # TODO: Implement Gumbel-Softmax
    pass

def saliency():
    pass

class ScResnet(nn.Module):
    def __init__(self, classifier='resnet18') -> None:
        super().__init__()
        self.resnet = create_model(classifier)

    def forward(self, x):
        x = self.resnet(x)
        return x

@register_model
def scresnet(pretrained=False, **kwargs):
    return ScResnet()