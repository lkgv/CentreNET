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

import VOC2012
from net03 import ConvNet
from utils import Configures
from loss import SmoothL1Loss

DEBUG = False
def weights_init(m):
    classname=m.__class__.__name__
    xavier = nn.init.xavier_normal
    if classname.find('Conv') != -1:
        xavier(m.weight.data)
        xavier(m.bias.data)

def init_net01(config):
    epoch = 0
    net = ConvNet(config)
    net = nn.DataParallel(net)
    if config('train', 'SNAPSHOT') != 'None':
        _, _, curepoch = os.path.basename(snapshot).split('_')
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, cursnapshot))
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
    config = Configures()

    seed = int(config('train', 'RANDOM_SEED'))
    base_lr = float(config('train', 'LR'))
    max_steps = int(config('data', 'SIZE'))
    alpha = float(config('train', 'ALPHA'))
    task = config('train', 'TASK') # 'seg'/'offset'
    batch_size = int(config('train', 'BATCH_SIZE'))
    epochnum = int(config('train', 'NUM_EPOCH'))
    milestone = int(config('train', 'MILESTONE'))
    gamma = float(config('train', 'GAMMA'))

    os.environ["CUDA_VISIBLE_DEVICES"] = config('train', 'WORKERS')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    models_path = os.path.abspath(os.path.expanduser('models'))
    os.makedirs(models_path, exist_ok=True)

    net, starting_epoch = init_net01(config=config)
    net.apply(weights_init)

    voc_loader = VOC2012.Loader(configure=config)
    train_loader = voc_loader()

    optimizer = optim.Adam(net.parameters(), lr=base_lr)
    scheduler = MultiStepLR(optimizer,
                            milestones=[x * milestone for x in range(1, 100)],
                            gamma=gamma)
    cls_criterion = nn.BCEWithLogitsLoss()
    ''' Losses tested for offsetmap
    seg_criterion = nn.NLLLoss2d()
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    d2_loss = (lambda a, b:
                torch.sum(
                    torch.sqrt(
                        torch.pow(a[:, 0, :, :] - b[:, 0, :, :], 2)
                        + torch.pow(a[:, 1, :, :] - b[:, 1, :, :], 2))))
    '''
    smthL1_criterion = nn.SmoothL1Loss()
    nll_criterion = nn.NLLLoss2d()

    curepoch = 0
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

            optimizer.zero_grad()

            if task == 'seg':
                out_seg = net(x, func='seg')
                loss = 100 * torch.abs(nll_criterion(out_seg, y_seg))
                epoch_losses.append(loss.data[0])

                status = '[{0}] Segmentation; loss:{1:0.6f}/{2:0.6f}, LR:{3:0.8f}'.format(
                    epoch + 1,
                    loss.data[0],
                    np.mean(epoch_losses),
                    scheduler.get_lr()[0])

            elif task == 'offset':
                out_cls, out = net(x, func='all')
                cls_loss = torch.abs(cls_criterion(out_cls, y_cls))
                ins_loss = torch.abs(smthL1_criterion(out, y))
                loss = ins_loss + alpha * cls_loss

                epoch_losses.append(loss.data[0])
                epoch_cls_losses.append(cls_loss.data[0])
                epoch_ins_losses.append(ins_loss.data[0])

                status = '[{0}] loss:{1:0.4f}/{2:0.4f},cls:{3:0.4f}/{4:0.4f},ins:{5:0.4f}/{6:0.4f} LR:{7:0.6f}'.format(
                    epoch + 1,
                    loss.data[0],
                    np.mean(epoch_losses),
                    cls_loss.data[0],
                    np.mean(epoch_cls_losses),
                    ins_loss.data[0],
                    np.mean(epoch_ins_losses),
                    scheduler.get_lr()[0])

            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["net03", task, str(epoch + 1)])))

if __name__ == '__main__':
    train()
