import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import variable
from torch_deform_conv.layers import ConvOffset2D

DEBUG = False

class NoShrunkenTransition(nn.Module):
    """ NoShrunkenTransition
    is the new transistion module for densenet,
    to prevent redundant pooling for image segmentation.

    ---
    'newpool' is the new pooling layer with no
    size shrunk for replacing original pooling layer.
    """
    def __init__(self, oritransistion):
        super(NoShrunkenTransition, self).__init__()
        for name, layer in oritransistion.named_children():
            if name == 'pool':
                newpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0,
                                       ceil_mode=False, count_include_pad=False)
                self.add_module('pool', newpool)
            else:
                self.add_module(name, layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class Dense121FeatureNet(nn.Module):
    def __init__(self):
        super(Dense121FeatureNet, self).__init__()
        basenet = models.densenet121(pretrained=True).features

        for name, layer in basenet.named_children():
            if 'transition' in name:
                # print(name)
                # print(layer)
                new_transition = NoShrunkenTransition(layer)
                self.add_module(name, new_transition)
            else:
                self.add_module(name, layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class Vgg16FeatureNet(nn.Module):
    def __init__(self):
        super(Vgg16FeatureNet, self).__init__()
        basenet = models.vgg16_bn(pretrained=True).features

        for name, layer in basenet.named_children():
            if name not in ('23', '33', '43'):
                self.add_module(name, layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class OffsetNet(nn.Module):
    def __init__(self, inchannel):
        super(OffsetNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, inchannel * 2, 3, padding=2, dilation=2)
        self.bn11 = nn.BatchNorm2d(inchannel * 2)

        # self.offset12 = ConvOffset2D(inchannel // 2)
        self.conv12 = nn.Conv2d(inchannel * 2, inchannel * 2, 3, padding=2, dilation=2)
        self.bn12 = nn.BatchNorm2d(inchannel * 2)

        self.conv13 = nn.Conv2d(inchannel * 2, inchannel * 2, 3, padding=2, dilation=2)
        self.bn13 = nn.BatchNorm2d(inchannel * 2)

        self.conv21 = nn.Conv2d(inchannel * 2, inchannel, 1, padding=1)
        self.bn21 = nn.BatchNorm2d(inchannel)

        self.conv22 = nn.Conv2d(inchannel, 2, 1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        # x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv13(x))
        x = self.bn13(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = self.conv22(x)

        if DEBUG:
            print('X size',x.size())
        return x

class ClassNet(nn.Module):
    def __init__(self, inchannel, nclass):
        super(ClassNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, 2048, 7, padding=6, dilation=2)
        self.bn11 = nn.BatchNorm2d(2048)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.lin1 = nn.Linear(2048, 2048)
        self.drop1 = nn.Dropout(0.5)

        self.lin2 = nn.Linear(2048, nclass)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = self.avgpool(x)
        x = x.squeeze(3).squeeze(2)

        x = F.relu(self.lin1(x))
        x = self.drop1(x)

        x = self.lin2(x)

        return x

class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        nclass = int(config('data', 'NUM_CLASSES'))

        self.features = Vgg16FeatureNet() # Dense121FeatureNet()

        inchannel = 512
        self.classifier = ClassNet(inchannel, 21)
        self.offset = OffsetNet(inchannel)

        self.upspl_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upspl_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.classifier = ClassNet(inchannel, nclass)
        # self.offset = OffsetNet(inchannel)


    def forward(self, x, func):
        featuremap = self.features(x)
        if DEBUG:
            print('feature: ', featuremap.size())

        if func == 'cls':
            classes = self.classifier(featuremap)
            return classes

        elif func == 'offset':
            offsetmap = self.offset(featuremap)
            offsetmap = self.upspl_1(offsetmap)
            offsetmap = self.upspl_2(offsetmap)
            if DEBUG:
                print('offset: ', offsetmap.size())
            return offsetmap

