import torch
import torch.nn as nn
from models.decoder import Decoder, MILADecoder
from models.encoder import resnet50
# from mod import AblatedNet
from models.resnet_encoder import ResNet50, ResNet18
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
from models.deeplabv3_encoder import DeepLabv3


class StaticTaskModel(nn.Module):
    ''' This class contains the implementation of a static model for a single task
            Args:
                enocder: A integer indicating the embedding size.
                decoder: A integer indicating the size of output dimension.
    '''

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.atten = attention
        # self.version = version

    def forward(self, input):
        # [b, c, h, w]
        print(input.shape)
        encoder_ftrs = self.encoder(input)
        # print(encoder_ftrs.shape) torch.Size([8, 256, 16, 32])
        segmentation_pred = self.decoder(encoder_ftrs)
        return segmentation_pred

    # function to log the weights of the model
    # def log_weights(self, step):
    #     # log the weights of the encoder
    #     writer.add_histogram("weights/MILADecoder/conv1/weight", model.conv1.weight.data, step)
    #     writer.add_histogram("weights/MILADecoder/conv1/bias", model.conv1.bias.data, step)
    #     writer.add_histogram("weights/MILADecoder/conv2/weight", model.conv2.weight.data, step)
    #     writer.add_histogram("weights/MILADecoder/conv2/bias", model.conv2.bias.data, step)
    #     writer.add_histogram("weights/MILADecoder/fc1/weight", model.fc1.weight.data, step)
    #     writer.add_histogram("weights/MILADecoder/fc1/bias", model.fc1.bias.data, step)
    #     writer.add_histogram("weights/MILADecoder/fc2/weight", model.fc2.weight.data, step)
    #     writer.add_histogram("weights/MILADecoder/fc2/bias", model.fc2.bias.data, step)



if __name__ == "__main__":
    num_classes = 19
    prev_sblock_kf = 0
    L = 5
    INPUT_DIM = 512
    dec_seg = MILADecoder(input_dim=256)
    # dec_seg = Decoder(input_dim=INPUT_DIM)
    # net = AblatedNet(c_in=3, c_out_seg=3)
    # encoder_slow = ResNet50(19, 3)
    # encoder_slow = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # encoder_slow = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=19)
    # encoder_fast = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    encoder_slow = DeepLabv3('resnet50').to('cpu')
    encoder_fast = DeepLabv3('resnet18').to('cpu')
    # encoder_fast = ResNet18(19, 3)
    batch = 8
    T = 3
    channels = 3
    height = 128
    width = 256
    input_tensor = torch.autograd.Variable(torch.rand(batch, T, channels, height, width))
    # semantic_pred = net(input_tensor)
    model = MTL_model(encoder_slow, encoder_fast, dec_seg, 5, version='mila')
    semantic_pred_slow = model(input_tensor)
    # print(semantic_pred.shape)
