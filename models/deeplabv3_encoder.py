import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_encoder import ResNet50
# from resnet_encoder import ResNet50
from torchvision import models
from tensorboardX import SummaryWriter
from torchvision.models import resnet50, resnet18, resnet101, resnet34, resnet152
class ResNet(nn.Module):
    def __init__(self, version='resnet50', in_channels=3, conv1_out=64):
        super(ResNet, self).__init__()
        if version == 'resnet50':
            self.resnet = resnet50(pretrained=True)
        elif version == 'resnet18':
            self.resnet = resnet18(pretrained=True)
        elif version == 'resnet101':
            self.resnet = resnet101(pretrained=True)
        elif version == 'resnet34':
            self.resnet = resnet34(pretrained=True)
        else:
            # version == 'resnet152':
            self.resnet = resnet152(pretrained=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        return x


class ASSP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASSP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               padding=0,
                               dilation=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=6,
                               dilation=6,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=12,
                               dilation=12,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=18,
                               dilation=18,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.conv5 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.convf = nn.Conv2d(in_channels=out_channels * 5,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        x5 = self.adapool(x)
        x5 = self.conv5(x5)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=tuple(x4.shape[-2:]), mode='bilinear')

        # print (x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # channels first
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)

        return x

class DeepLabv3(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(DeepLabv3, self).__init__()
        self.backbone = backbone
        # self.nc = nc
        print(backbone)
        self.resnet = ResNet(backbone)
        if backbone == 'resnet50':
            self.assp = ASSP(in_channels=1024)
        elif backbone == 'resnet18':
            self.assp = ASSP(in_channels=256)
        elif backbone == 'resnet101':
            self.assp = ASSP(in_channels=1024)
        else:
            self.assp = ASSP(in_channels=512)

        self.conv = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=1, stride=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        '''
        for module in self.assp.modules():
          if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
          elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=0.01, std=0.02)
            nn.init.constant_(module.bias, 0)
    
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        '''
    def forward(self, x):
        _, _, h, w = x.shape
        x = self.resnet(x)
        x = self.assp(x)
        # print(x.shape)
        # if self.backbone == 'resnet18':
        #     x = self.conv(x)
        # x = self.conv(x)
        # x = F.interpolate(x, size=(h, w), mode='bilinear')  # scale_factor = 16, mode='bilinear')
        return x

