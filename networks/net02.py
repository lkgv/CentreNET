import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

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
        middle0 = x.contiguous()
        tagged = ['5', '12', '22', '42']

        for name, layer in self.named_children():
            x = layer(x)
            if name == '5':
                middle1 = x.contiguous()
            elif name == '12':
                middle2 = x.contiguous()
            elif name == '22':
                middle3 = x.contiguous()
            elif name == '42':
                middle4 = x.contiguous()

        return x, middle0, middle1, middle2, middle3, middle4

class Convert4xNet(nn.Module):
    def __init__(self, inchannel):
        super(Convert4xNet, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 130, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(130)

        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(130, 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)

        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = self.pool2(x)

        return x

class Convert2xNet(nn.Module):
    def __init__(self, inchannel):
        super(Convert2xNet, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 130, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(130)

        self.conv3 = nn.Conv2d(130, 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)

        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = self.pool2(x)

        return x

class Convert1xNet(nn.Module):
    def __init__(self, inchannel):
        super(Convert1xNet, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 130, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(130)

        self.conv3 = nn.Conv2d(130, 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        return x

class OffsetNet(nn.Module):
    def __init__(self, inchannel):
        super(OffsetNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, 2, 3, padding=1)

        self.upspl_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upspl_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv11(x)

        fin = x.contiguous()

        classfeature = fin # torch.cat((fon, fin), 1)

        x = self.upspl_1(x)
        x = self.upspl_2(x)

        if DEBUG:
            print('X size',x.size())
            print('in offset, x size:', x.size())
        return x, classfeature #  * 512.0 - 128.0

class SegNet(nn.Module):
    def __init__(self, inchannel, nclass):
        super(SegNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, nclass, 3, padding=1)

        self.upspl_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upspl_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv11(x)

        x = self.upspl_1(x)
        x = self.upspl_2(x)
        return x

class ClassNet(nn.Module):
    def __init__(self, inchannel, nclass):
        super(ClassNet, self).__init__()

        self.conv11 = nn.Conv2d(inchannel, 128, 3, padding=2, dilation=2)
        self.bn11 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.lin1 = nn.Linear(128, nclass)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = self.avgpool(x)
        x = x.squeeze(3).squeeze(2)

        x = self.lin1(x)

        return x

class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()

        nclass = int(config('data', 'NUM_CLASSES'))
        inchannel = 512

        self.seg = SegNet(inchannel, nclass)
        self.features = Vgg16FeatureNet() # Dense121FeatureNet()
        self.classifier = ClassNet(2, nclass)
        self.offset = OffsetNet(20)
        self.converter0 = Convert4xNet(3)
        self.converter1 = Convert4xNet(64)
        self.converter2 = Convert2xNet(128)
        self.converter3 = Convert1xNet(256)
        self.converter4 = Convert1xNet(512)

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

