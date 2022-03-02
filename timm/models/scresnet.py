import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .factory import create_model
from .registry import register_model
from .helpers import build_model_with_cfg
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchsummary import summary


# __all__ = ['ScResnet']  # model_registry will add each entrypoint fn to this

default_cfg = {
    'num_classes': 1000,
    'input_size': (4, 224, 224), 
    'interpolation': 'bilinear',
    # below only needed with using with prefetcher.
    # 'crop_pct': 1.0,
    # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
    # 'mean': (0, 0, 0, 0),
    # 'std': (1, 1, 1, 1)
}
model_cfg = {
    'SC_layers': [
        # in_chan, out_chan, kernel_size, stride, activation
        [ 1,  16, 3, 2, 'relu6'],
        [ 16, 32, 3, 2, 'relu6'],
        [ 32, 64, 3, 2, 'relu6']
    ],
    'downsample_size': (1, 64, 64),
    'original_size': (3, 224, 224),
    'num_classes': default_cfg['num_classes'],
    'classifier': 'resnet18'
}

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
        self.fc = nn.Linear(c * h * w, 6)

    def forward(self, x):
        # x: (batch, n_crop=1, chan=1, H, W)
        batch_size = x.shape[0]
        n = x.shape[1]
        x = x.view(batch_size * n, x.shape[2], x.shape[3], x.shape[4])
        x = self.blocks(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x.view(batch_size, 6)
        x = F.gumbel_softmax(x, dim=1)
        return x

class ScResnet(nn.Module):
    def __init__(self, original_size, downsample_size, SC_layers,
                 classifier='resnet18', num_classes=1000, in_chans=4) -> None:
        super().__init__()
        self.down_size = downsample_size
        self.orig_size = original_size
        self.num_classes=num_classes

        self.salience_map = ScLayer(SC_layers, self.down_size)
        print(downsample_size)
        print(summary(self.salience_map, (2,1,self.down_size[0],self.down_size[1],self.down_size[2])))

        self.resnet = create_model(classifier, in_chans=original_size[0])

        self.is_training = True
        
        self.image_cnt = 0
        self.enable_image_save = True


    def forward(self, x):
        # x -> (batch, 5_crop(5)+orig(1)+grey(1), chan=3, H, W)
        # print(f'the shape of input img={x.shape}')
        x = torch.permute(x, (1, 0, 2, 3, 4))
        # x -> (chan=4, batch, n_crop, H, W)
        assert x.shape[0] == 7
        x_sc = x[0].unsqueeze(dim=0)     # (n_crop=1, batch, chan=3, H, W)
        x_cls = x[1:]   # (n_crop=6, batch, chan=3, H, W)
        x_sc = torch.permute(x_sc, (1, 0, 2, 3, 4)) # (batch, n_crop=1, chan=3, H, W)
        x_sc = x_sc[:, :, :self.down_size[0], :self.down_size[1], :self.down_size[2]]   # remove the padded region (batch, n_crop=1, chan=1, H, W)
        assert x_sc.shape == (x_sc.shape[0], 1, 1, self.down_size[1], self.down_size[2])
        x_cls = torch.permute(x_cls, (1, 0, 2, 3, 4)) # (batch, n_crop, chan=3, H, W)
        
        trans = transforms.ToPILImage()
        if self.image_cnt <= 2 and self.enable_image_save:
            for j in range(10):
                greys = x_sc[j]
                for i in range(greys.shape[0]):
                    grey = trans(greys[i])
                    grey_name = f'check/img-{self.image_cnt*10 + j}-grey-{i}.png'
                    grey.save(grey_name)

                colors = x_cls[j]
                for i in range(colors.shape[0]):
                    color = trans(colors[i])
                    color_name = f'check/img-{j}-color-{i}.png'
                    color.save(color_name)

        x_sc = self.salience_map(x_sc)  # (batch, n_crop)
        if self.training:
            # print('train')
            x_sc = x_sc.view(x_sc.shape[0], x_sc.shape[1], 1, 1, 1)
            x_cls = x_cls * x_sc
            x_cls = torch.sum(x_cls, dim=1)
        else:
            # print('eval'))
            with torch.no_grad():
                # print(torch.sum(x_sc, dim=0) / x_sc.shape[0])
                x_sc = torch.argmax(x_sc, dim=1)    # [batch_size,]
                # print(x_sc)
                x_sc = x_sc.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x_sc = x_sc.repeat(1, 1, self.orig_size[0], self.orig_size[1], self.orig_size[2])
                x_cls = torch.gather(x_cls, dim=1, index=x_sc)
                x_cls = torch.squeeze(x_cls)
                
                if self.image_cnt <= 2 and self.enable_image_save:
                    for i in range(10):
                        image = trans(x_cls[i])
                        image_name = f'check/img-{self.image_cnt*10 + i}-EvalSelected.png'
                        image.save(image_name)
        self.image_cnt += 1
        assert x_cls.shape[1:] == self.orig_size
        x = self.resnet(x_cls)
        return x

@register_model
def scresnet(**kwargs):
    pretrained = False
    return build_model_with_cfg(ScResnet, 'scresnet', pretrained, default_cfg, **model_cfg)