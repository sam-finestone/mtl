import torch
import torch.nn as nn
from decoder import *
from encoder import resnet50
from mod import AblatedNet
from resnet_encoder import ResNet50, ResNetFast

class MTL_model(nn.Module):
    ''' This class contains the implementation of the slowfast model.
    https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py
            Args:
                enocder: A integer indicating the embedding size.
                decoder: A integer indicating the size of output dimension.
                number_frames: A integer indicating the hidden size of rnn.
    '''
    def __init__(self, encoder_slow, encoder_fast, seg_decoder, K=5, attention=True):
        super().__init__()
        self.encoder_slow = encoder_slow
        self.encoder_fast = encoder_fast
        self.seg_decoder = seg_decoder
        # self.depth_decoder = depth_decoder
        self.K = 5
        self.atten = attention

    def forward(self, input, lateral=0, prev_sblock_kf=0):
        keyframes = input[::self.K, :, :, :]
        non_keyframes = input[::2, :, :, :]
        # print(keyframes.shape)
        # print(non_keyframes.shape)
        # initialise the lateral list of tensors
        # pass the keyframes through the SlowPath way
        # slow_encoder = self.encoder(keyframes, keyframe=True)
        # fast_encoder = self.encoder(non_keyframes, keyframe=False)
        # torch.Size([1, 2048, 4, 8])
        enc_slow_ftrs = self.encoder_slow(keyframes)
        print(enc_slow_ftrs.shape)
        # reshape the encoder_slower to concatenate same dim
        enc_slow_ftrs = enc_slow_ftrs.contiguous().view(8, -1, 4, 8)
        # torch.Size([8, 256, 4, 8])
        # print(enc_slow_ftrs.shape)
        enc_fast_ftrs = self.encoder_fast(non_keyframes)
        print(enc_fast_ftrs.shape)
        # x_slow_pred = Decoder(x_slow)
        enc_combined_ftrs = torch.cat([enc_fast_ftrs, enc_slow_ftrs], dim=1)
        print(enc_combined_ftrs.shape)
        slow_output = self.seg_decoder(enc_combined_ftrs)
        # fast_encoder = self.encoder(non_keyframes)
        print(slow_output.shape)
        # slow_decoder = self.decoder(slow_encoder)
        # input_fused = torch.cat([slow_encoder, fast_encoder], dim=1)
        # x_fast_pred = self.decoder(input_fused)



if __name__ == "__main__":
    num_classes = 101
    a = torch.zeros(1, 16, 4, 32, 64)
    b = torch.zeros(1, 64, 4, 32, 64)
    c = torch.zeros(1, 128, 4, 16, 32)
    d = torch.zeros(1, 256, 4, 8, 16)
    lateral = [a, b, c, d]
    prev_sblock_kf = 0
    L = 5
    INPUT_DIM = 512
    OUTPUT_DIM = [(3, 128, 256)]
    dec_seg = Decoder(1280)
    net = AblatedNet(c_in=3, c_out_seg=3)
    encoder_slow = ResNet50(19, 3)
    encoder_fast = ResNetFast(19, 3)
    input_tensor = torch.autograd.Variable(torch.rand(16, 3, 128, 256))
    # semantic_pred = net(input_tensor)
    model = MTL_model(encoder_slow, encoder_fast, dec_seg, 5)

    semantic_pred = model(input_tensor)
    print(semantic_pred.shape)