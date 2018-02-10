import numpy as np
import torch

from PIL import Image
import numpy as np

Index = { 'aeroplane' : 1,
          'bicycle' : 2,
          'bird' : 3,
          'boat' : 4,
          'bottle' : 5,
          'bus' : 6,
          'car' : 7,
          'cat' : 8,
          'chair' : 9,
          'cow' : 10,
          'diningtable' : 11,
          'dog' : 12,
          'horse' : 13,
          'motorbike' : 14,
          'person' : 15,
          'pottedplant' : 16,
          'sheep' : 17,
          'sofa' : 18,
          'train' : 19,
          'tvmonitor' : 20 }

class target_transform:

    def __call__(self, offset, annotation):
        cls = np.zeros(21)
        cls[0] = 1.

        if isinstance(annotation['object'], list):
            objs = annotation['object']
            for obj in objs:
                cls[Index[obj['name']]] += 1.
        else:
            obj = annotation['object']
            cls[Index[obj['name']]] += 1.

        return offset, cls
