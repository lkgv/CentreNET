import os
import xmltodict
import numpy as np
from numpy import array_equal as arreq
import skimage as skm
import skimage.io as sio

__all__ = ['get_centers', 'ann_to_offset']


def checkpath(p):
    path = os.path.expanduser(p)
    path = os.path.abspath(path)
    return path

listpath = '~/data/VOC2012/VOCdevkit/VOC2012/' + \
        'ImageSets/Segmentation/trainval.txt'
listpath = checkpath(listpath)
datapath = '~/data/VOC2012/VOCdevkit/VOC2012/'
datapath = checkpath(datapath)

imgpath = os.path.join(datapath, 'JPEGImages')
clspath = os.path.join(datapath, 'SegmentationClass')
objpath = os.path.join(datapath, 'SegmentationObject')
offsetpath = os.path.join(datapath, 'SegmentationOffset')
annotationpath = os.path.join(datapath, 'Annotations')

cutpath = os.path.join(datapath, 'Pointer')

edge = np.array([224, 224, 192])
bg = np.array([0, 0, 0])


def threebyte2int(x, y, z):
    ret = str((x << 16) | (y << 8) | z)
    return ret


def trip2int(x):
    return threebyte2int(*x)

edgel = trip2int(edge)
bgl = trip2int(bg)

CLASS = {threebyte2int(0, 0, 0): 'background',
         threebyte2int(224, 224, 192): 'edge',
         threebyte2int(128, 0, 0): 'aeroplane',
         threebyte2int(0, 128, 0): 'bicycle',
         threebyte2int(128, 128, 0): 'bird',
         threebyte2int(0, 0, 128): 'boat',
         threebyte2int(128, 0, 128): 'bottle',
         threebyte2int(0, 128, 128): 'bus',
         threebyte2int(128, 128, 128): 'car',
         threebyte2int(64, 0, 0): 'cat',
         threebyte2int(192, 0, 0): 'chair',
         threebyte2int(64, 128, 0): 'cow',
         threebyte2int(192, 128, 0): 'diningtable',
         threebyte2int(64, 0, 128): 'dog',
         threebyte2int(192, 0, 128): 'horse',
         threebyte2int(64, 128, 128): 'motorbike',
         threebyte2int(192, 128, 128): 'person',
         threebyte2int(0, 64, 0): 'pottedplant',
         threebyte2int(128, 64, 0): 'sheep',
         threebyte2int(0, 192, 0): 'sofa',
         threebyte2int(128, 192, 0): 'train',
         threebyte2int(0, 64, 128): 'tvmonitor'}
CLS_NAME = list(CLASS.values())


def get_centers(img):
    height, width, _ = img.shape

    c2i = (lambda i, j: trip2int(img[i, j]))
    pixelsum = {}
    pixelsum = {
        c2i(i, j): ((i, j, 1) +
                    (pixelsum[c2i(i, j)]
                     if c2i(i, j) in pixelsum
                     else (0., 0., 0.)))
        for i in range(height) for j in range(width)
    }

    pixelavg = {
        key: ((lambda x:
               np.array([float(x[0]) / x[2],
                         float(x[1]) / x[2]]))
              (pixelsum[key]))
        for key in pixelsum.keys()
    }
    return pixelavg


def cut_windows(shape, boxes):
    height, width = int(shape['height']), int(shape['width'])
    centX, centY = (height // 2, width // 2)
    windows = [[(0, 0), (256, 256)],
               [(height - 256, 0), (height, 256)],
               [(0, width - 256), (256, width)],
               [(height - 256, width - 256), (height, width)],
               [(centX - 128, centY - 128), (centX + 128, centY + 128)]]

    windowlist = []
    for window in windows:
        current = {
            'ymin': window[0][0],
            'xmin': window[0][1],
            'ymax': window[1][0],
            'xmax': window[1][1],
            'bbox': []
        }

        for box in boxes:
            ymin = max(box['ymin'], window[0][0])
            xmin = max(box['xmin'], window[0][1])
            ymax = min(box['ymax'], window[1][0])
            xmax = min(box['xmax'], window[1][1])

            iarea = (xmax - xmin) * (ymax - ymin)
            barea = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])

            if float(iarea) / float(barea) >= 2.0 / 3.0:
                boxinwindow = {'ymin': int(ymin),
                               'xmin': int(xmin),
                               'ymax': int(ymax),
                               'xmax': int(xmax),
                               'name': box['name']}
                current['bbox'].append(boxinwindow)

        if len(current['bbox']) > 0:
            windowlist.append(current)
    return windowlist


def get_window_bbox(ann):
    cookbox = (lambda obj:
               {'xmax': int(obj['bndbox']['xmax']),
                'xmin': int(obj['bndbox']['xmin']),
                'ymax': int(obj['bndbox']['ymax']),
                'ymin': int(obj['bndbox']['ymin']),
                'name': obj['name']})

    if 'object' in ann:
        objtype = str(type(ann['object']))
        if objtype == "<class 'list'>":
            boxes = [cookbox(item) for item in ann['object']]
        elif objtype == "<class 'collections.OrderedDict'>":
            boxes = [cookbox(ann['object'])]
        else:
            raise TypeError('Invalid type for objects in annotation')
        windows = cut_windows(ann['size'], boxes)
    else:
        windows = None

    return windows


def get_window_offset(image, segment, offset, windows):
    ret = []
    for window in windows:
        # print('window:', window)
        offsetmask = np.zeros((256, 256, 2))
        clsmask = np.zeros((256, 256, 3)).astype(int)
        classes = []

        for bbox in window['bbox']:
            name = bbox['name']
            iarea = 0.
            barea = float((bbox['ymax'] - bbox['ymin']) *
                          (bbox['xmax'] - bbox['xmin']))
            if (not isinstance(bbox['xmax'], int) or
                    not isinstance(bbox['ymax'], int) or
                    not isinstance(bbox['xmin'], int) or
                    not isinstance(bbox['ymin'], int)):
                print(bbox)
            for i in range(bbox['ymin'], bbox['ymax']):
                for j in range(bbox['xmin'], bbox['xmax']):

                    cls = segment[i, j]
                    centy, centx = offset[i, j] + (i, j)
                    inbox = (bbox['ymin'] <= centy and centy < bbox['ymax'] and
                             bbox['xmin'] <= centx and centx < bbox['xmax'])

                    if CLASS[trip2int(cls)] == name and inbox:
                        iarea += 1.
            if iarea / barea >= 1. / 4.:
                bbox_2 = {'name': bbox['name'],
                          'ymin': bbox['ymin'] - window['ymin'],
                          'xmin': bbox['xmin'] - window['xmin'],
                          'ymax': bbox['ymax'] - window['ymin'],
                          'xmax': bbox['xmax'] - window['xmin']}
                classes.append(bbox_2)

                for i in range(bbox['ymin'], bbox['ymax']):
                    for j in range(bbox['xmin'], bbox['xmax']):

                        cls = segment[i, j]
                        centy, centx = offset[i, j] + (i, j)

                        if (CLASS[trip2int(cls)] == name and
                                bbox['ymin'] <= centy and
                                centy < bbox['ymax'] and
                                bbox['xmin'] <= centx and
                                centx < bbox['xmax']):
                            offsetmask[i - window['ymin'],
                                       j - window['xmin']] = offset[i, j]
                            clsmask[i - window['ymin'],
                                    j - window['xmin']
                                    ] = np.around(segment[i, j]).astype(int)

        # pack image, mask, class for proper window
        ymin, xmin = window['ymin'], window['xmin']
        ymax, xmax = window['ymax'], window['xmax']
        # print('ymin: {}, xmin: {}, ymax: {}, xmax: {}'
        #       .format(ymin, xmin, ymax, xmax))
        if len(classes) > 0:
            result = {'image': image[ymin: ymax, xmin: xmax],
                      'offsetmask': offsetmask,
                      'clsmask': clsmask,
                      'class': classes}
            ret.append(result)
    return ret


def ann_to_offset(img):
    height, width, _ = img.shape
    pixelavg = get_centers(img)
    offsetmap = np.ones((height, width, 2))

    for i in range(height):
        for j in range(width):

            if not np.array_equal(img[i, j], edge) \
               and not np.array_equal(img[i, j], bg):

                target = pixelavg[trip2int(img[i, j])]
                offsetmap[i, j] = target - np.array([float(i), float(j)])
                norm2 = np.linalg.norm(offsetmap[i, j], 2)
                if norm2 > 0:
                    offsetmap[i, j] = offsetmap[i, j] / norm2
                else:
                    offsetmap[i, j] = np.zeros(2)

            else:
                offsetmap[i, j] = np.zeros(2)

    return offsetmap


def main():
    with open(listpath, 'r') as f:
        imglist = f.read().split('\n')[:-1]

    # imglist = ['2011_003145']

    def checkdir(path):
        if os.path.isfile(path):
            print('current path {} is a file instead of dir'.format(path))
            raise TypeError('There have been a file in appointed path!')
        elif os.path.isdir(path):
            pass
        else:
            os.mkdir(path)
    checkdir(os.path.join(cutpath, 'JPEGImages'))
    checkdir(os.path.join(cutpath, 'SegmentationClass'))
    checkdir(os.path.join(cutpath, 'SegmentationOffset'))
    checkdir(os.path.join(cutpath, 'Annotations'))
    checkdir(os.path.join(cutpath, 'ImageSets'))
    checkdir(os.path.join(cutpath, 'ImageSets', 'Segmentation'))

    saved_list = []
    trashed_list = []

    for i, index in enumerate(imglist):
        print('The ' + str(i) + 'st image, index is ' + index)

        ''' TASK_1: GETTING OFFSETMAP FOREACH IMAGE
        img = sio.imread(os.path.join(datapath, index + '.png'))
        out = ann_to_offset(img)

        np.save(os.path.join(resultpath, index), out)
        '''

        ''' TASK_2
        '''

        img = sio.imread(os.path.join(imgpath, index + '.jpg'))
        x, y, _ = img.shape
        if min(x, y) < 256:
            trashed_list.append(index)
            continue

        seg = sio.imread(os.path.join(clspath, index + '.png'))
        obj = sio.imread(os.path.join(objpath, index + '.png'))
        # offset = np.load(os.path.join(offsetpath, index + '.npy'))
        offset = ann_to_offset(obj)
        with open(os.path.join(annotationpath, index + '.xml')) as f:
            annotation = xmltodict.parse(f.read())['annotation']

        bbox = get_window_bbox(annotation)

        windows = get_window_offset(img, seg, offset, windows=bbox)
        for k, window in enumerate(windows):
            curimg = window['image']
            curoff = window['offsetmask']
            curcls = window['clsmask']
            curn = window['class']

            saved_list.append('{}_{}'.format(index, k))

            sio.imsave(os.path.join(cutpath,
                                    'JPEGImages',
                                    '{}_{}.jpg'.format(index, k)),
                       curimg)
            sio.imsave(os.path.join(cutpath,
                                    'SegmentationClass',
                                    '{}_{}.png'.format(index, k)),
                       curcls)
            np.save(os.path.join(cutpath,
                                 'SegmentationOffset',
                                 '{}_{}.npy'.format(index, k)),
                    curoff)
            if len(curn) < 1:
                raise ValueError('no ojbect in the window when {0}_{1}'
                                 .format(index, k))
            else:
                cls = [tmp['name'] for tmp in curn]
                count = {tmp: cls.count(tmp) for tmp in CLS_NAME}
                obj = {'annotation': {'object': curn, 'count': count}}

            xml = xmltodict.unparse(obj)
            with open(os.path.join(cutpath,
                                   'Annotations',
                                   '{}_{}.xml'.format(index, k)), 'w') as f:
                f.write(xml)

    with open(os.path.join(cutpath,
                           'ImageSets',
                           'Segmentation',
                           'trainval.txt'), 'w') as f:
        f.write('\n'.join(saved_list))

    with open(os.path.join(cutpath, 'trashed.txt'), 'w') as f:
        f.write('\n'.join(trashed_list))


if __name__ == '__main__':
    main()
