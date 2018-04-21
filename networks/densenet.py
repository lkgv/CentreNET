import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

DEBUG = False

class Features(nn.Module):
    def __init__(self, orig_module):
        super(Features, self).__init__()
        self.orig = orig_module
    
    def forward(self, x):
        x = self.orig.conv0(x)

        feature0 = x.contiguous()

        x = self.orig.norm0(x)
        x = self.orig.relu0(x)
        x = self.orig.pool0(x)
        x = self.orig.denseblock1(x)
        
        feature1 = x.contiguous()

        x = self.orig.transition1(x)
        x = self.orig.denseblock2(x)

        feature2 = x.contiguous()

        x = self.orig.transition2(x)
        x = self.orig.denseblock3(x)

        feature3 = x.contiguous()

        x = self.orig.transition3(x)
        x = self.orig.denseblock4(x)
        x = F.relu(self.orig.norm5(x))

        return x, feature0, feature1, feature2, feature3


class Classifier(nn.Module):
    def __init__(self, orig_module, num_classes = 22):
        super(Classifier, self).__init__()
        self.orig = orig_module
        self.convert = nn.Linear(1000, num_classes, bias=True)
    
    def forward(self, x):
        x = F.relu(self.orig(x.view(-1, 1664)))
        x = self.convert(x)
        return x


class Segmenter(nn.Module):
    def __init__(self, num_classes = 22):
        super(Segmenter, self).__init__()

        self.conv0 = nn.Conv2d(1664, 1000, (1,1))
        self.bn0 = nn.BatchNorm2d(1000, eps=1e-05, momentum=0.1, affine=True)
        self.relu0 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(1000, 640, 4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(1920, 512, 4, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False)
        
        self.deconv0_1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False)
        self.deconv0 = nn.ConvTranspose2d(768, 136, 8, stride=4, padding=2, bias=False)

        self.conv1 = nn.Conv2d(200, num_classes, (1, 1))

    def forward(self, x, feature0, feature1, feature2, feature3):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = F.relu(self.deconv3(x))
        x = torch.cat((x, feature3), 1).contiguous()
        x = F.relu(self.deconv2(x))
        x = torch.cat((x, feature2), 1).contiguous()
        x = F.relu(self.deconv1(x))
        x = torch.cat((x, feature1), 1).contiguous()

        x = F.relu(self.deconv0(x))
        feature0 = F.relu(self.deconv0_1(feature0))
        x = torch.cat((x, feature0), 1).contiguous()

        x = self.conv1(x)

        return x


class ConvNet(nn.Module):
    def __init__(self, config, task='train'):
        super(ConvNet, self).__init__()

        archi_name = config('network', 'ARCHITECTURE')
        pretrained = config('network', 'PRETRAINED')
        base_net = models.__dict__[archi_name](pretrained)
        self.features = Features(base_net.features)
        num_classes = config('data', 'NUM_CLASSES')
        self.classifier = Classifier(base_net.classifier, num_classes)
        self.segmenter = Segmenter(num_classes)
        self.transfer_weight()

    def transfer_weight(self):
        self.segmenter.conv0.weight.data.copy_(self.classifier.orig.weight.data)

    def forward(self, x, function):
        feat, feature0, feature1, feature2, feature3 = self.features(x)

        if 'class' in function:
            class_feat = feat.contiguous()
            class_feat = F.avg_pool2d(feat, kernel_size=8, stride=1)
            class_result = self.classifier(class_feat)
        else:
            class_result = None

        if 'segment' in function:
            segment_result = self.segmenter(
                feat, 
                feature0, feature1, feature2, feature3)
        else:
            segment_result = None

        if class_result is not None and segment_result is not None:
            return class_result, segment_result
        elif class_result is not None:
            return class_result
        elif segment_result is not None:
            return segment_result

