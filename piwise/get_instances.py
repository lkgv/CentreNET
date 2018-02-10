from PIL import Image
import skimage as skm
from skimage import io as sio
import numpy as np
import xmltodict

import os
import torch
from torch.utils.data import Dataset

class Instance(Dataset):
    def __init__(self, root, mode='trainval', input_transform=None, target_transform=None):
        root = os.path.abspath(os.path.expanduser(root))
        self.images_root = os.path.join(root, 'JPEGImages')
        self.segments_root = os.path.join(root, 'SegmentationClass')
        self.annotations_root = os.path.join(root, 'Annotations')
        self.offset_root = os.path.join(root, 'SegmentationOffset')

        list_file = os.path.join(root,
                                 'ImageSets',
                                 'Segmentation',
                                 '{}.txt'.format(mode))
        with open(list_file) as f:
            items = f.read().split('\n')
        self.imglist = [item for item in items if len(item) > 0]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fileid = self.imglist[index]

        '''
        imgfile = os.path.join(self.images_root,
                               '{}.jpg'.format(fileid))
        with open(imgfile, 'rb') as f:
            image = Image.open(f).convert('RGB')
        segmentfile = os.path.join(self.segments_root,
                                   '{}.png'.format(fileid))
        with open(segmentfile, 'rb') as f:
            segmentation = Image.open(f).convert('P')
        '''

        imgfile = os.path.join(self.images_root,
                               '{}.jpg'.format(fileid))
        with open(imgfile, 'rb') as f:
            image = Image.open(f).convert('RGB')
        offsetfile = os.path.join(self.offset_root,
                                  '{}.npy'.format(fileid))
        offset = np.load(offsetfile)
        annotationfile = os.path.join(self.annotations_root,
                                      '{}.xml'.format(fileid))
        with open(annotationfile) as f:
            tmp = xmltodict.parse(f.read())
            annotation = tmp['annotation']

        if self.input_transform is not None:
            image = self.input_transform(image)

        if self.target_transform is not None:
            offset, annotation = self.target_transform(
                offset, annotation)
        '''
        if self.target_transform is not None:
            annotation, segmentation = self.target_transform(
                annotation, segmentation)
        '''
        # if image is not None:
        #    print('image:', type(image))
        # image.cuda()
        # print('image: ', image.size())
        annotation = torch.Tensor(annotation)
        offset = torch.Tensor(offset).transpose(0, 2).transpose(1, 2)#[:, :256, :256]
        #print('Image:', image.size())
        #print('anno:', annotation.size())
        #print('offset:', offset.size())
        # a, b, c = image.size()
        # v = image[:, :min(b,300), :min(c, 100)]
        return image, offset, annotation #np.random.randn(3, 45, 70)#offset #image #, #annotation, offset # segmentation

    def __len__(self):
        return len(self.imglist)

