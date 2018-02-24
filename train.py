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
from net01 import ConvNet
from utils import Configures


DEBUG = False


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

def init_net01(config, snapshot = None):
    epoch = 0
    net = ConvNet(config)
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
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

    os.environ["CUDA_VISIBLE_DEVICES"] = config('train', 'WORKERS')
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    seed = int(config('train', 'RANDOM_SEED'))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    models_path = os.path.abspath(os.path.expanduser('models'))
    os.makedirs(models_path, exist_ok=True)

    net, starting_epoch = init_net01(config=config)

    voc_loader = VOC2012.Loader(configure=config)

    # train_loader, class_weights, n_images = voc_loader(data_path, batch_size, len(eval(gpu)))
    train_loader, class_weights, n_images = voc_loader()

    optimizer = optim.Adam(net.parameters(), lr=float(config('train', 'LR')))

    scheduler = MultiStepLR(optimizer, milestones=[x * 10 for x in range(1, 10)], gamma=0.5)

    max_steps = 5428

    alpha = float(config('train', 'ALPHA'))

    curepoch = 0

    for epoch in range(starting_epoch, starting_epoch + int(config('train', 'NUM_EPOCH'))):
        curepoch += 1

        class_weights.cuda()
        seg_criterion = nn.NLLLoss2d()
        cls_criterion = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        d2_loss = (lambda a, b:
                   torch.sum(
                       torch.sqrt(
                           torch.pow(a[:, 0, :, :] - b[:, 0, :, :], 2)
                           + torch.pow(a[:, 1, :, :] - b[:, 1, :, :], 2))))
        # seg_criterion = nn.NLLLoss2d()
        # cls_criterion = nn.BCEWithLogitsLoss()

        batch_size = int(config('train', 'BATCH_SIZE'))
        epoch_losses = []

        train_iterator = tqdm(train_loader, total=max_steps // batch_size + 1)
        steps = 0

        net.train()
        for x, y, y_cls in train_iterator:
            steps += batch_size
            optimizer.zero_grad()
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()

            out, out_cls = None, None
            if curepoch < 4 and curepoch % 2 == 1:
                out_cls = net(x, func='cls')
                loss = cls_criterion(out_cls, y_cls)
            else:
                out = net(x, func='offset')
                out_cls = net(x, func='cls')
                # loss = mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
                loss = d2_loss(out, y) / batch_size + alpha * cls_criterion(out_cls, y_cls)

            epoch_losses.append(loss.data[0])

            if DEBUG:
                print('x:', x.size())
                print('y:', y.size())
                print('y_cls:', y_cls.size())
                print('out:', out.size())
                print('out_cls:', out_cls.size())
                print('out:', out.shape)
                print('y:', y.shape)

            status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {3:0.7f}'.format(
                epoch + 1,
                loss.data[0],
                np.mean(epoch_losses),
                scheduler.get_lr()[0])
            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["net01", str(epoch + 1)])))
        train_loss = np.mean(epoch_losses)


if __name__ == '__main__':
    train()
