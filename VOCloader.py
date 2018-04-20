import torch

from PIL import Image
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from VOCdataset.dataset import VOC12
from VOCdataset.get_instances import Instance
from VOCdataset.transform import Relabel, ToLabel, Colorize
from VOCdataset.transform_instance import target_transform

class Loader:
    def __init__(self, configure, task='train'):
        self._config = configure
        self._task = task
        self.NUM_CHANNELS = int(self._config('data', 'NUM_CHANNELS'))
        self.NUM_CLASSES = int(self._config('data', 'NUM_CLASSES'))

        self.color_transform = Colorize()
        self.image_transform = ToPILImage()
        self.input_transform = Compose([
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        '''
        self.target_transform = Compose([
            CenterCrop(256),
            ToLabel(),
            Relabel(255, self.NUM_CLASSES - 1),
        ])
        '''
        self.target_transform = target_transform()

    def __call__(self):
        if self._task == 'train':
            workers = len(self._config('train', 'WORKERS').split(','))
            batch_size = int(self._config('train','BATCH_SIZE'))
        elif self._task == 'val':
            workers = 1
            batch_size = 1

        loader = DataLoader(
            # VOC12(self._config('data', 'DATA_DIR'), self.input_transform, self.target_transform),
            Instance(self._config('data', 'DATA_DIR'),
                     input_transform=self.input_transform,
                     target_transform=self.target_transform),
            num_workers=workers,
            batch_size=batch_size,
            shuffle=True)

        return loader
