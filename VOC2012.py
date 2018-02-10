import torch

from PIL import Image
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from piwise.dataset import VOC12
from piwise.get_instances import Instance
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.transform_instance import target_transform

class Loader:
    def __init__(self, configure):
        self._config = configure
        self.NUM_CHANNELS = int(self._config('data', 'NUM_CHANNELS'))
        self.NUM_CLASSES = int(self._config('data', 'NUM_CLASSES'))

        self.color_transform = Colorize()
        self.image_transform = ToPILImage()
        self.input_transform = Compose([
            # CenterCrop(256),
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

        self.weight = torch.ones(21)
        self.weight[0] = 0

    def __call__(self):
        workers = list(map(int, self._config('train', 'WORKERS').split(',')))

        loader = DataLoader(
            # VOC12(self._config('data', 'DATA_DIR'), self.input_transform, self.target_transform),
            Instance(self._config('data', 'DATA_DIR'),
                     input_transform=self.input_transform,
                     target_transform=self.target_transform),
            num_workers=len(workers),
            batch_size=int(self._config('train','BATCH_SIZE')),
            shuffle=True)

        return loader, self.weight, self.NUM_CLASSES
