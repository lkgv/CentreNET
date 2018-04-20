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

import VOCloader
from networks.base import ConvNet
from utils import Configures, checkdir

configFile = 'val.yml'


def init_net(net, config):
    net = nn.DataParallel(net)
    model_name = config('val', 'MODEL')
    _, _, epoch = model_name.split('_')

    model_path = os.path.join('models', model_name)
    net.load_state_dict(torch.load(model_path))
    net = net.cuda()
    
    return net, epoch


def main():
    expdir = os.path.abspath(os.path.expanduser('exp'))
    checkdir(expdir)

    config = Configures(configFile)
    os.environ["CUDA_VISIBLE_DEVICES"] = config('val', 'WORKER')
    max_steps = config('data', 'SIZE')

    net = ConvNet(config, task='val')
    net, starting_epoch = init_net(net, config)

    voc_loader = VOCloader.Loader(configure=config, task='val')
    train_loader = voc_loader()
    train_iterator = tqdm(train_loader, total=max_steps)

    count = 0
    net.eval()

    for x, y, y_cls in train_iterator:

        x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()

        out_cls, out = net(x, func='all')

        count += 1
        
        outdir = os.path.join(expdir, str(count).zfill(6))
        checkdir(outdir)
        name_x = os.path.join(outdir, 'X.npy')
        name_y = os.path.join(outdir, 'y.npy')
        name_out = os.path.join(outdir, 'out.npy')

        xs = x.data[0].cpu().transpose(0,2).transpose(0,1).numpy()
        np.save(name_x, xs)
        ys =  y.data[0].cpu().numpy()
        np.save(name_y, ys)
        outs =  out.data[0].cpu().numpy()
        np.save(name_out, outs)


if __name__ == '__main__':
    main()
