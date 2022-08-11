import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet_encoder import Block
from models.attention.attention import LocalContextAttentionBlock
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, SynchronizedBatchNorm2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

class SE_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ILA_Layer(nn.Module):
    '''
    This class contains the inter-frame local attention (ILA)
    which accounts for motion by finding local attention
    weights in inter-frames
    Agrs:
        L: the window size
    '''
    def __init__(self, input_dim, output_dim, window_size):
        super().__init__()
        self.L = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.kH = kH
        # self.kW = kW
        # self.prev_se_block = previous_se_block
        self.conv1 = nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.shared_weight = nn.Parameter(self.conv1.weight)

    @staticmethod
    def f_similar(ft, fk, l):
        n, c, h, w = ft.size()  # (N, inter_channels, H, W)
        pad = (l // 2, l // 2)
        ft = ft.permute(0, 2, 3, 1).contiguous()
        print(ft.shape)
        ft = ft.view(n * h * w, 1, c)
        print(ft.shape)

        fk = F.unfold(fk, kernel_size=(l, l), stride=1, padding=pad)
        fk = fk.contiguous().view(n, c, l * l, h * w)
        fk = fk.permute(0, 3, 1, 2).contiguous()
        fk = fk.view(n * h * w, c, l * l)

        out = torch.matmul(ft, fk)
        out = out.view(n, h, w, l * l)
        return out

    @staticmethod
    def f_weighting(ft, fk, l):
        n, c, h, w = ft.size()  # (N, inter_channels, H, W)
        pad = (l // 2, l // 2)
        ft = F.unfold(ft, kernel_size=(l, l), stride=1, padding=pad)
        ft = ft.permute(0, 2, 1).contiguous()
        ft = ft.view(n * h * w, c, l * l)

        fk = fk.view(n * h * w, l * l, 1)

        out = torch.matmul(ft, fk)
        out = out.squeeze(-1)
        out = out.view(n, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

    def forward(self, ft, fk):
        # outputs = []
        key = F.conv2d(ft, self.shared_weight)
        query = F.conv2d(fk, self.shared_weight)
        print(key.shape)
        print(query.shape)
        # x3 = self.conv3(x)
        # weight = self.f_similar(key, query, self.L)
        # print(weight.shape)
        weight = F.softmax(weight, -1)
        out = self.f_weighting(ft, weight, self.L)

        return out


class Decoder(nn.Module):
    ''' This class contains the implementation of a basic decoder
        Args:
            input_dim: A integer indicating the size of input dimension
            of the concatenated encoded features
            n_layers: A integer indicating the feature layers in the cnns
        '''

    def __init__(self, input_dim, n_layers=[128], l=50, nc=19):
        super().__init__()
        self.input_dim = input_dim
        self.t_dim = 1
        self.L = l
        # SE block - input encoder dimension and reduction ratio
        self.se_layer = SE_Layer(self.input_dim, 16)
        # self.ila_layer = ILA_Layer(self.input_dim, self.input_dim, self.L)
        self.conv1 = nn.Conv2d(self.input_dim, n_layers[0], kernel_size=(1, 1), stride=2)

        self.conv2 = nn.Conv2d(n_layers[0], nc, kernel_size=(1, 1), stride=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=(4, 1, 1), stride=(2, 1, 1))
        # self.conv3 = nn.Conv2d(16, 1, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # self.upsample = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        # self.up_trans = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)

    def forward(self, x):
        # print('Running decoder')
        print(x.shape)
        x = self.se_layer(x)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        x = self.upsample(x)
        print(x.shape)
        return x


class MILADecoder(nn.Module):
    ''' This class contains the implementation of the decoder in the MILA paper
        which uses their inter-frame attention module (ILA)
        Args:
            input_dim: A integer indicating the size of input dimension
            of the concatenated encoded features
            n_layers: A integer indicating the feature layers in the cnns
        '''

    def __init__(self, input_dim, n_layers=[128], l=50, nc=19):
        super().__init__()
        self.input_dim = input_dim
        self.t_dim = 1
        self.L = l
        # SE block - input encoder dimension and reduction ratio
        self.se_layer = SE_Layer(self.input_dim, 16)
        self.local_attention_block = LocalContextAttentionBlock(in_channels=self.input_dim, out_channels=128, kernel_size=3)
        self.conv1 = nn.Conv2d(self.input_dim, n_layers[0], kernel_size=(1, 1), stride=2)
        self.conv2 = nn.Conv2d(n_layers[0], nc, kernel_size=(1, 1), stride=2)
        self.upsample = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)

    def forward(self, slow_pathway, fast_se_feature):
        # print('Running decoder')
        print(slow_pathway.shape)
        se_output_slow = self.se_layer(slow_pathway)
        # se_output_fast = self.se_layer(slow_pathway)
        print(slow_pathway.shape)
        ila_output_slow = self.local_attention_block(fast_se_feature, slow_pathway)
        print(ila_output_slow.shape)
        # we concatenate ila layer with se_block
        slow_concat = torch.cat([ila_output_slow, se_output_slow], dim=1)
        # Feed outputs through conv layers for each task
        x = self.conv1(slow_concat)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        slow_pred = self.upsample(x)
        # print(x.shape)
        return slow_pred, se_output_slow


class Decoder(nn.Module):
    ''' This class contains the implementation of a basic decoder
        Args:
            input_dim: A integer indicating the size of input dimension
            of the concatenated encoded features
            n_layers: A integer indicating the feature layers in the cnns
        '''

    def __init__(self, input_dim, n_layers=[128], l=50, nc=19):
        super().__init__()
        self.input_dim = input_dim
        self.t_dim = 1
        self.L = l
        # SE block - input encoder dimension and reduction ratio
        self.se_layer = SE_Layer(self.input_dim, 16)
        # self.ila_layer = ILA_Layer(self.input_dim, self.input_dim, self.L)
        self.conv1 = nn.Conv2d(self.input_dim, n_layers[0], kernel_size=(1, 1), stride=2)

        self.conv2 = nn.Conv2d(n_layers[0], nc, kernel_size=(1, 1), stride=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=(4, 1, 1), stride=(2, 1, 1))
        # self.conv3 = nn.Conv2d(16, 1, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # self.upsample = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        # self.up_trans = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)

    def forward(self, x):
        # print('Running decoder')
        print(x.shape)
        x = self.se_layer(x)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        x = self.upsample(x)
        print(x.shape)
        return x



class SegDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm):
        super(SegDecoder, self).__init__()
        # if backbone == "resnet" or backbone == "drn":
        #     low_level_inplanes = 256
        self.conv1 = nn.Conv2d(input_dim, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = torch.cat((x, low_level_feat), dim=1)
        x = F.interpolate(x, size=(256, 512), mode="bilinear", align_corners=True)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class DepthDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm):
        super(DepthDecoder, self).__init__()
        # if backbone == "resnet" or backbone == "drn":
        #     low_level_inplanes = 256
        self.conv1 = nn.Conv2d(input_dim, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = torch.cat((x, low_level_feat), dim=1)
        x = F.interpolate(x, size=(256, 512), mode="bilinear", align_corners=True)
        x = self.last_conv(x)
        x = self.softplus(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()