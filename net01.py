import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import variable
from torch_deform_conv.layers import ConvOffset2D

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
        basenet = models.densenet121(pretrained=False).features

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
        basenet = models.vgg16(pretrained=False).features

        for name, layer in basenet.named_children():
            if name not in ('23', '30'):
                self.add_module(name, layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

class OffsetNet(nn.Module):
    def __init__(self, inchannel):
        super(OffsetNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, inchannel // 2, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(inchannel // 2)

        # self.offset12 = ConvOffset2D(inchannel // 2)
        self.conv12 = nn.Conv2d(inchannel // 2, inchannel // 2, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(inchannel // 2)

        self.conv21 = nn.Conv2d(inchannel // 2, 2, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(2)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        # x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv21(x))
        print('X size',x.size())
        return x

class ClassNet(nn.Module):
    def __init__(self, inchannel, nclass):
        super(ClassNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d(2)

        self.conv21 = nn.Conv2d(512, 512, 2, padding=1)
        self.bn21 = nn.BatchNorm2d(512)

        self.conv22 = nn.Conv2d(512, nclass, 2, padding=0)
        self.avg2 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = self.avgpool(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = F.softmax(self.avg2(self.conv22(x)))

        x = x.squeeze(3)
        x = x.squeeze(2)

        return x



class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        nclass = int(config('data', 'NUM_CLASSES'))

        self.features = Vgg16FeatureNet() # Dense121FeatureNet()

        inchannel = 1024
        self.classifier = ClassNet(1024, 21)
        self.offset = OffsetNet(1024)
        # self.classifier = ClassNet(inchannel, nclass)
        # self.offset = OffsetNet(inchannel)


    def forward(self, x):
        featuremap = self.features(x)
        print('feature: ', featuremap.size())
        offsetmap = self.offset(featuremap)
        print('offset: ', offsetmap.size())
        classes = self.classifier(featuremap)

        return offsetmap, classes


