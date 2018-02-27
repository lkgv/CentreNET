import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

import skimage as ski
import skimage.io as sio
from tqdm import tqdm
import os
import click
import logging
import numpy as np

import VOC2012
from net01 import ConvNet
from utils import Configures

model_name = 'net01_12'

model_path = os.path.abspath(os.path.expanduser('models'))
model_dir = os.path.join(model_path, model_name)

configFile = os.path.abspath(os.path.expanduser('val.ini'))

def init_net01(config, snapshot = None):
    epoch = 0
    net = ConvNet(config)
    net = nn.DataParallel(net)
    net = net.cuda()
    return net, epoch

def checkdir(path):
    if os.path.isfile(path):
        print('current path {} is a file instead of dir'.format(path))
        raise TypeError('There have been a file in appointed path!')
    elif os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

def main():
    config = Configures(configFile)

    os.environ["CUDA_VISIBLE_DEVICES"] = config('train', 'WORKERS')
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    seed = int(config('train', 'RANDOM_SEED'))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = int(config('train', 'BATCH_SIZE'))

    models_path = os.path.abspath(os.path.expanduser('models'))
    os.makedirs(models_path, exist_ok=True)

    net, starting_epoch = init_net01(config=config)

    voc_loader = VOC2012.Loader(configure=config)

    # train_loader, class_weights, n_images = voc_loader(data_path, batch_size, len(eval(gpu)))
    train_loader, class_weights, n_images = voc_loader()

    max_steps = 5428

    train_iterator = tqdm(train_loader, total=max_steps // batch_size + 1)

    count = 0

    for x, y, y_cls in train_iterator:

        x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()

        out_cls, out = net(x, func='all')

        for i in range(batch_size):
            count += 1
            expdir = os.path.abspath(os.path.expanduser('exp'))
            outdir = os.path.join(expdir, str(count).zfill(6))
            checkdir(expdir)
            checkdir(outdir)

            name_x = os.path.join(outdir, 'X.npy')
            name_y = os.path.join(outdir, 'y.npy')
            name_out = os.path.join(outdir, 'out.npy')

            xs = x.data[i].cpu().transpose(0,2).transpose(0,1).numpy()
            np.save(name_x, xs)
            ys =  y.data[i].cpu().numpy()
            np.save(name_y, ys)
            outs =  out.data[i].cpu().numpy()
            np.save(name_out, outs)

if __name__ == '__main__':
    main()
