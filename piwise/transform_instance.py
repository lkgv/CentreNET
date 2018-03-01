import numpy as np
import torch

from PIL import Image
import numpy as np

Index = { 'aeroplane' : 2,
          'bicycle' : 3,
          'bird' : 4,
          'boat' : 5,
          'bottle' : 6,
          'bus' : 7,
          'car' : 8,
          'cat' : 9,
          'chair' : 10,
          'cow' : 11,
          'diningtable' : 12,
          'dog' : 13,
          'horse' : 14,
          'motorbike' : 15,
          'person' : 16,
          'pottedplant' : 17,
          'sheep' : 18,
          'sofa' : 19,
          'train' : 20,
          'tvmonitor' : 21 }

class target_transform:

    def __call__(self, offset, annotation):
        cls = np.zeros(22)
        cls[0] = 1.

        if isinstance(annotation['object'], list):
            objs = annotation['object']
            for obj in objs:
                cls[Index[obj['name']]] += 1.
        else:
            obj = annotation['object']
            cls[Index[obj['name']]] += 1.

        return offset, cls
