""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch.torch_version import TorchVersion
import torch.utils.data as data
import os
import torch
import logging
from torchvision import transforms

from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .parsers import create_parser

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map=None,
            load_bytes=False,
            transform=None,
            target_transform=None,
    ):
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            parser=None,
            split='train',
            is_training=False,
            batch_size=None,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert parser is not None
        if isinstance(parser, str):
            self.parser = create_parser(
                parser, root=root, split=split, is_training=is_training,
                batch_size=batch_size, repeats=repeats, download=download)
        else:
            self.parser = parser
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.parser:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.parser, '__len__'):
            return len(self.parser)
        else:
            return 0

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)


'''
Additional Dataset type for SalienceMap+Classification model
'''
class SalienceImageDataset(ImageDataset):
    def __init__(self, root, parser=None, class_map=None, load_bytes=False, transform=None, target_transform=None):
        super().__init__(root, parser=parser, class_map=class_map, load_bytes=load_bytes, transform=transform, target_transform=target_transform)
        self.small_size = 64
        self.large_size = 224
        self.downsize = transforms.Compose([
            transforms.Resize((self.small_size, self.small_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad([0, 0, self.large_size-self.small_size, self.large_size-self.small_size]),
            transforms.ToTensor()
        ])
        self.original = transforms.Compose([
            transforms.Resize((self.large_size, self.large_size)),
            transforms.ToTensor()
        ])
        self.downsize_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.FiveCrop(self.large_size),   # outputs PIL img
            transforms.Lambda(lambda images: [transforms.Resize(self.small_size)(img) for img in images]),
            transforms.Lambda(lambda images: [transforms.Grayscale(num_output_channels=1)(img) for img in images]),
            transforms.Lambda(lambda images: [transforms.Pad([0, 0, self.large_size-self.small_size, self.large_size-self.small_size])(img) for img in images]),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) # returns a 4D tensor
        ])
        self.original_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.FiveCrop(self.large_size),   # outputs PIL img
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) # returns a 4D tensor
        ])
        self.enable_img_save = True

    def __getitem__(self, index):
        if type(self.transform.transforms[-1]) == transforms.Normalize:
            self.transform.transforms = self.transform.transforms[:-1]
        
        img, target = super().__getitem__(index)
        if img.shape == (6, 4, self.large_size, self.large_size):
            return img, target
        
        trans = transforms.ToPILImage()
        img = trans(img)
        if index % 10000 == 1 and self.enable_img_save:
            image_name = f'check/img{index}-toPIL.png'
            img.save(image_name)
        
        ds_nocrop = self.downsize(img)
        ds_nocrop = torch.unsqueeze(ds_nocrop, 0)   # torch.Size([1, 1, 224, 224])

        ori_nocrop = self.original(img)
        ori_nocrop = torch.unsqueeze(ori_nocrop, 0) # torch.Size([1, 3, 224, 224])
        
        downsize_crop = self.downsize_transform(img)    # torch.Size([5, 1, 224, 224])
        downsize_crop = torch.cat((ds_nocrop, downsize_crop), dim=0)    # torch.Size([6, 1, 224, 224])

        original_crop = self.original_transform(img)    # torch.Size([5, 3, 224, 224])
        original_crop = torch.cat((ori_nocrop, original_crop), dim=0)   # torch.Size([6, 3, 224, 224])

        downsize_crop = torch.permute(downsize_crop, (1, 0, 2, 3))
        original_crop = torch.permute(original_crop, (1, 0, 2, 3))
        img = torch.cat((original_crop,downsize_crop), dim=0)
        img = torch.permute(img, (1, 0, 2, 3))
        
        if index % 10000 == 1 and self.enable_img_save:
            assert img.shape == (6, 4, self.large_size, self.large_size)
            for i in range(img.shape[0]):
                color= trans(img[i][:3])
                grey = trans(img[i][3])
                color_name = f'check/img{index}-color-{i}.png'
                grey_name = f'check/img{index}-grey-{i}.png'
                color.save(color_name)
                grey.save(grey_name)

        return img, target