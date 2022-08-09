import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['resnet50', 'resnet101', 'resnet152', 'resnet200']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x

class SlowFast(nn.Module):
    ''' This class contains the implementation of the slowfast model.
    https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py
            Args:
                embedding_dim: A integer indicating the embedding size.
                output_dim: A integer indicating the size of output dimension.
                hidden_dim: A integer indicating the hidden size of rnn.
                n_layers: A integer indicating the number of layers in rnn.
                dropout: A float indicating the dropout.
    '''
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], fast_blocks=[8, 16, 32, 64], slow_blocks=[64, 128, 256, 512],
                 keyframe=5,
                 seg_classes=19):
        super(SlowFast, self).__init__()
        self.nb_classes = seg_classes
        self.K = keyframe
        self.in_channels = layers[0]
        self.fast_conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.fast_bn1 = nn.BatchNorm2d(self.in_channels)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_fast2 = self.get_resnet_slow(block, fast_blocks[0], layers[0])
        self.res_fast3 = self.get_resnet_slow(block, fast_blocks[1], layers[1], stride=2)
        self.res_fast4 = self.get_resnet_slow(block, fast_blocks[2], layers[2], stride=2)
        self.res_fast5 = self.get_resnet_slow(block, fast_blocks[3], layers[3], stride=2)
        self.fast_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        assert len(slow_blocks) == len(layers) == 4
        self.slow_conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.slow_bn1 = nn.BatchNorm2d(self.in_channels)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_slow2 = self.get_resnet_slow(block, slow_blocks[0], layers[0])
        self.res_slow3 = self.get_resnet_slow(block, slow_blocks[1], layers[1], stride=2)
        self.res_slow4 = self.get_resnet_slow(block, slow_blocks[2], layers[2], stride=2)
        self.res_slow5 = self.get_resnet_slow(block, slow_blocks[3], layers[3], stride=2)
        self.slow_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(self.in_channels, output_dim)

    def forward(self, input_frame, keyframe):
        # fast, lateral = self.FastPath(input[:, :, ::2, :, :])
        # keyframes = input[:, :, ::self.K, :, :]
        # non_keyframes = input[:, :, ::2, :, :]
        # fast - torch.Size([1, 256])
        # print(input_frame.shape)
        if keyframe:
            output = self.SlowPath(input_frame)
        else:
            output = self.FastPath(input_frame)
        # slow - torch.Size([1, 2048])
        # slow = self.SlowPath(keyframes, lateral)
        # x = torch.cat([slow, fast], dim=1)
        # # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # pred_semantics = self.fc1(x)
        return output

    def SlowPath(self, x):
        print(x.shape)
        x = self.slow_conv1(x)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = self.res_slow2(x)
        x = self.res_slow3(x)
        x = self.res_slow4(x)
        x = self.res_slow5(x)
        # x = self.slow_avgpool(x)
        print(x.shape)
        # h = x.view(x.shape[0], -1)
        # x = self.fc(h)
        return x

    def FastPath(self, input):
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        # lateral_p = self.lateral_p1(pool1)
        # lateral.append(lateral_p)
        res2 = self.res_fast2(pool1)
        # lateral_res2 = self.lateral_res2(res2)
        # lateral.append(lateral_res2)
        res3 = self.res_fast3(res2)
        # lateral_res3 = self.lateral_res3(res3)
        # lateral.append(lateral_res3)
        res4 = self.res_fast4(res3)
        # lateral_res4 = self.lateral_res4(res4)
        # lateral.append(lateral_res4)
        res5 = self.res_fast5(res4)
        # print(res5.shape)
        # x = self.fast_avgpool(res5)
        # h = x.view(x.shape[0], -1)
        return res5

    def get_resnet_fast(self, block, n_blocks, channels, stride=1):
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)

    def get_resnet_slow(self, block, n_blocks, channels, stride=1):
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], [8, 16, 32, 64], [64, 128, 256, 512], 5, 19)
    return model


# def resnet101(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = SlowFast(Bottleneck, [3, 4, 23, 3], 5, 19, **kwargs)
#     return model
#
#
# def resnet152(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
#     return model
#
#
# def resnet200(**kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
#     return model


if __name__ == "__main__":
    num_classes = 101
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 64, 128, 256))
    model = resnet50()
    output_slow, output_fast = model(input_tensor)
    print(output_slow.size())