import numpy as np
import skimage as ski

__all__ = ['png2mask']

threebyte2int = lambda x, y, z: str((x << 16) | (y << 8) | z)
trip2int = lambda x: threebyte2int(*x)

edge = np.array([224, 224, 192])
bg = np.array([0, 0, 0])

edgel = trip2int(edge)
bgl = trip2int(bg)

CLASS = { threebyte2int(0, 0, 0): 'background',
          threebyte2int(224, 224, 192): 'edge',
          threebyte2int(128, 0, 0) : 'aeroplane',
          threebyte2int(0, 128, 0) : 'bicycle',
          threebyte2int(128, 128, 0) : 'bird',
          threebyte2int(0, 0, 128) : 'boat',
          threebyte2int(128, 0, 128) : 'bottle',
          threebyte2int(0, 128, 128) : 'bus',
          threebyte2int(128, 128, 128) : 'car',
          threebyte2int(64, 0, 0) : 'cat',
          threebyte2int(192, 0, 0) : 'chair',
          threebyte2int(64, 128, 0) : 'cow',
          threebyte2int(192, 128, 0) : 'diningtable',
          threebyte2int(64, 0, 128) : 'dog',
          threebyte2int(192, 0, 128) : 'horse',
          threebyte2int(64, 128, 128) : 'motorbike',
          threebyte2int(192, 128, 128) : 'person',
          threebyte2int(0, 64, 0) : 'pottedplant',
          threebyte2int(128, 64, 0) : 'sheep',
          threebyte2int(0, 192, 0) : 'sofa',
          threebyte2int(128, 192, 0) : 'train',
          threebyte2int(0, 64, 128) : 'tvmonitor' }

CLS_VEC = [(0, 0, 0), (224, 224, 192), (128, 0, 0),
           (0, 128, 0), (128, 128, 0), (0, 0, 128),
           (128, 0, 128), (0, 128, 128), (128, 128, 128),
           (64, 0, 0), (192, 0, 0), (64, 128, 0),
           (192, 128, 0), (64, 0, 128), (192, 0, 128),
           (64, 128, 128), (192, 128, 128), (0, 64, 0),
           (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
CLS_VEC = [threebyte2int(*x) for x in CLS_VEC]

def png2mask(img):
    img = np.array(img).astype(int)
    x, y, _ = img.shape
    out = np.zeros((x, y)).astype(int)

    for i in range(x):
        for j in range(y):
            cur = threebyte2int(*img[i, j])
            if cur in CLS_VEC:
                out[i, j] = CLS_VEC.index(cur)
    return out


