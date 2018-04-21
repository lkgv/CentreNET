import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import click
import logging
import numpy as np

import VOCloader
import networks.densenet
from utils import Configures

ConvNet = networks.densenet.ConvNet
DEBUG = False


def weights_init(m):
    classname=m.__class__.__name__
    xavier = nn.init.kaiming_uniform
    constant = nn.init.constant
    # if classname.find('Conv2d') != -1 or 
    if classname.find('Linear') != -1:
        xavier(m.weight.data)
        constant(m.bias.data, 0.1)


def init_net01(config):
    epoch = 0
    net = ConvNet(config)
    net = nn.DataParallel(net)
    net.apply(weights_init)
    snapshot = config('train', 'SNAPSHOT')
    if snapshot != 'None':
        _, _, curepoch = os.path.basename(snapshot).split('_')
        net.load_state_dict(torch.load(os.path.join('models', snapshot)))
        logging.info("Snapshot for epoch {} loaded from {}".format(curepoch, snapshot))
    net = net.cuda()
    return net, epoch

'''
    To follow this training routine you need a DataLoader that yields the tuples of the following format:
    (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
    x - batch of input images,
    y - batch of groung truth seg maps,
    y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
    y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
'''
def train():
    config = Configures(cfg='train.yml')

    seed = int(config('train', 'RANDOM_SEED'))
    base_lr = float(config('train', 'LR'))
    max_steps = int(config('data', 'SIZE'))
    alpha = float(config('train', 'ALPHA'))
    task = config('train', 'TASK') # 'seg'/'offset'
    batch_size = int(config('train', 'BATCH_SIZE'))
    level1 = config('train', 'level1')
    level2 = config('train', 'level2')
    level3 = config('train', 'level3')
    epochnum = level1 + level2 + level3
    milestone = int(config('train', 'MILESTONE'))
    gamma = float(config('train', 'GAMMA'))

    os.environ["CUDA_VISIBLE_DEVICES"] = config('train', 'WORKERS')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    models_path = os.path.abspath(os.path.expanduser('models'))
    os.makedirs(models_path, exist_ok=True)
    net, starting_epoch = init_net01(config=config)

    voc_loader = VOCloader.Loader(configure=config)
    train_loader = voc_loader()

    optimizer = optim.Adam(net.parameters(), lr=base_lr)
    seg_optimizer = optim.Adam(net.module.segmenter.parameters(), lr=base_lr)
    scheduler = MultiStepLR(optimizer,
                            milestones=[x * milestone for x in range(1, 1000)],
                            gamma=gamma)
    cls_criterion = nn.BCEWithLogitsLoss()
    ''' Losses tested for offsetmap
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    d2_loss = (lambda a, b:
                torch.sum(
                    torch.sqrt(
                        torch.pow(a[:, 0, :, :] - b[:, 0, :, :], 2)
                        + torch.pow(a[:, 1, :, :] - b[:, 1, :, :], 2))))
    '''
    smthL1_criterion = nn.SmoothL1Loss()
    seg_criterion = nn.NLLLoss2d()
    nll_criterion = nn.NLLLoss2d()

    curepoch = 0
    global_loss = []
    
    for epoch in range(starting_epoch, starting_epoch + epochnum):
        curepoch += 1

        epoch_losses = []
        epoch_ins_losses = []
        epoch_cls_losses = []

        train_iterator = tqdm(train_loader, total=max_steps // batch_size + 1)
        steps = 0

        net.train()
        for x, y, y_cls, y_seg in train_iterator:
            steps += batch_size
            x = Variable(x.cuda())
            y = Variable(y.cuda())
            y_cls = Variable(y_cls.cuda())
            y_seg = Variable(y_seg.cuda())

            # optimizer.zero_grad()

            if curepoch <= level1:
                optimizer.zero_grad()
                out_cls = net(x, function='classification')
                loss = torch.abs(cls_criterion(out_cls, y_cls))
                epoch_losses.append(loss.data[0])

                if curepoch == level1:
                    net.module.transfer_weight()

                status = '[{0}] Classification; loss:{1:0.6f}/{2:0.6f}, LR:{3:0.8f}'.format(
                    epoch + 1,
                    loss.data[0],
                    np.mean(epoch_losses),
                    scheduler.get_lr()[0])
                loss.backward()
                optimizer.step()
            
            elif curepoch <= level1 + level2:
                seg_optimizer.zero_grad()
                out_segment = net(x, function='segmentation')
                loss = torch.abs(seg_criterion(out_segment, y_seg))
                epoch_losses.append(loss.data[0])

                status = '[{0}] Segmentation; loss:{1:0.6f}/{2:0.6f}, LR:{3:0.8f}'.format(
                    epoch + 1,
                    loss.data[0],
                    np.mean(epoch_losses),
                    scheduler.get_lr()[0])
                loss.backward()
                seg_optimizer.step()            

            elif curepoch <= level1 + level2 + level3:
                optimizer.zero_grad()
                out_cls, out_segment = net(x, function='classification + segmentation')
                loss = alpha * torch.abs(cls_criterion(out_cls, y_cls)) + torch.abs(seg_criterion(out_segment, y_seg))
                epoch_losses.append(loss.data[0])

                status = '[{0}] Double; loss:{1:0.6f}/{2:0.6f}, LR:{3:0.8f}'.format(
                    epoch + 1,
                    loss.data[0],
                    np.mean(epoch_losses),
                    scheduler.get_lr()[0])
                loss.backward()
                optimizer.step()

            train_iterator.set_description(status)
            # loss.backward()
            # optimizer.step()
        if curepoch <= level1 or curepoch > level1 + level2:
            scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["dense", task, str(epoch + 1)])))
        
        global_loss.append((curepoch, loss.data[0]))

    with open('train.log', 'w') as f:
        f.write(str(global_loss))

if __name__ == '__main__':
    train()
