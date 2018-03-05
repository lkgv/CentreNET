import numpy as np
import os

from PIL import Image

import torch
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, '{0}{1}'.format(basename, extension))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(root, 'SegmentationClass')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, EXTENSIONS[0]), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, EXTENSIONS[1]), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        label_cls = torch.zeros(22)
        la, lb, lc = label.size()
        for i in range(la):
            for j in range(lb):
                for k in range(lc):
                    label_cls[label[i, j, k]] = 1

        return image, label, label_cls

    def __len__(self):
        return len(self.filenames)
