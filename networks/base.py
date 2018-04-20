import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

DEBUG = False

architectures = {
    'vgg11': models.vgg11_bn,
    'vgg13': models.vgg13_bn,
    'vgg16': models.vgg16_bn,
    'vgg19': models.vgg19_bn,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50
}


class ConvNet(nn.Module):
    def __init__(self, config, task='train'):
        super(ConvNet, self).__init__()

        archi = config('network', 'ARCHITECTURE')
        pretrained = config('network', 'PRETRAINED')
        if archi == 'vgg16':
            base_net = models.vgg16_bn(pretrained)
        elif archi == 'vgg19':
            base_net = models.vgg19_bn(pretrained)
        elif archi == ''
        base_net = models.densenet169(pretrained)
        self.features = base_net.features
        self.classifier = base_net.classifier


    def forward(self, x, func):
        featuremap, middle0, middle1, middle2, middle3, middle4 = self.features(x)
        skips = torch.cat((self.converter0(middle0),
                           self.converter1(middle1),
                           self.converter2(middle2),
                           self.converter3(middle3),
                           self.converter4(middle4)),
                          1).contiguous()

        if func == 'cls':
            classes = self.classifier(featuremap)
            return classes

        elif func == 'offset':
            offsetmap = self.offset(featuremap)
            if DEBUG:
                print('offset: ', offsetmap.size())
                print(offsetmap.size())
            return offsetmap

        elif func == 'all':
            offsetmap, classmap = self.offset(skips)
            classes = self.classifier(classmap)
            return classes, offsetmap

        elif func == 'seg':
            segmap = self.seg(featuremap)
            return segmap

