import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.resnet import resnet18
import timm

from .registry import register_model

# __all__ = ['ScResnet']  # model_registry will add each entrypoint fn to this

def gumbel_softmax(x):
    # TODO: Implement Gumbel-Softmax
    pass

def saliency():
    pass

@register_model
class ScResnet(nn.Module):
    def __init__(self, classifier='resnet18') -> None:
        super().__init__()
        self.resnet = timm.create_model(classifier)

    def forward(self, x):
        x = self.resnet(x)
        return x
        