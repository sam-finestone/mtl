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

class MultiTaskModel2(nn.Module):
    ''' This class contains the implementation of the slowfast model.
    https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py
            Args:
                enocder: A integer indicating the embedding size.
                decoder: A integer indicating the size of output dimension.
                number_frames: A integer indicating the hidden size of rnn.
    '''

    def __init__(self, backbone_encoder, task_decoders, keyframe_intervals=5, version='mila'):
        super().__init__()
        self.encoder_slow = encoder_slow
        self.encoder_fast = encoder_fast
        self.task_decoders = task_decoders
        self.K = keyframe_intervals
        # self.atten = attention
        self.conv = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.version = version
        self.fast_se_feature = torch.zeros((8, 256, 8, 16))


class MultiTaskModel(nn.Module):
    ''' This class contains the implementation of the slowfast model.
    https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py
            Args:
                enocder: A integer indicating the embedding size.
                decoder: A integer indicating the size of output dimension.
                number_frames: A integer indicating the hidden size of rnn.
    '''

    def __init__(self, encoder_slow, encoder_fast, list_decoders, k=5, version='mila', task_channels=[19, 3, 1]):
        super().__init__()
        self.encoder_slow = encoder_slow
        self.encoder_fast = encoder_fast
        self.decoders = list_decoders
        # self.depth_decoder = depth_decoder
        self.K = k
        # self.atten = attention
        self.conv = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.version = version
        self.fast_se_feature = torch.zeros((8, 256, 8, 16))

    def forward(self, input):
        # [b, t, c, h, w]
        # Get the keyframes and non-keyframes for slow/fast setting
        keyframes, non_keyframes, list_kf_indicies = self.get_keyframes(input)

        # change the dimensions of the input tensor for the deeplabv3 encoder and obtain encoder features
        enc_slow_ftrs, batch_size, t_dim_slow = self.run_encoder(keyframes, self.encoder_slow)
        enc_fast_ftrs, _, t_dim_fast = self.run_encoder(non_keyframes, self.encoder_fast)

        # torch tile to make the slow tensor same dim as fast on dim=0
        enc_slow_ftrs_tiled = enc_slow_ftrs.tile((2, 1, 1, 1)) # is there a better way of doing this?

        # Different ways of propagating temporal features
        if self.version == 'basic':
            enc_combined_ftrs = torch.cat([enc_fast_ftrs, enc_slow_ftrs_tiled], dim=1)
            enc_slow_ftrs = self.conv(enc_slow_ftrs)
            # run a decoder for each task
            task_slow_pred = []
            task_combin_pred = []
            task_predictions = []
            for task_decoder in self.decoders:
                # Reshape the output tensors to a [B, T, C, H, W]
                output_slow = self.reshape_output(task_decoder(enc_slow_ftrs), batch_size, t_dim_slow)
                output_fast = self.reshape_output(task_decoder(enc_combined_ftrs), batch_size, t_dim_fast)
                pred = self.stack_predictions(output_slow, output_fast, list_kf_indicies)
                task_slow_pred.append(output_slow)
                task_combin_pred.append(output_fast)
                task_predictions.append(pred)
        else:
            seg_slow_pred, se_output_slow = self.seg_decoder(enc_slow_ftrs, self.fast_se_feature)
            seg_fast_pred, se_output_fast = self.seg_decoder(enc_fast_ftrs, se_output_slow)
            self.fast_se_feature = se_output_fast

        # Reshape the output tensors to a [B, T, C, H, W]
        # output_seg_slow = self.reshape_output(seg_slow_pred, batch_size, t_dim_slow)
        # output_seg_fast = self.reshape_output(seg_fast_pred, batch_size, t_dim_fast)

        # want to stake the predicted frames back in order of the T
        # segmentation_pred = self.stack_predictions(output_seg_slow, output_seg_fast, list_kf_indicies)
        return task_predictions

    def get_keyframes(self, input):
        # input size dimension - [B, T, C, H, W]
        # create keyframes and non-keyframes using K
        T = input.shape[1]
        list_kf_indicies = [x for x in range(T) if x % self.K == 0]
        list_non_kf_indicies = list(set(range(T)) - set(list_kf_indicies))
        keyframes = input[:, list_kf_indicies, :, :, :]
        non_keyframes = input[:, list_non_kf_indicies, :, :, :]
        return keyframes, non_keyframes, list_kf_indicies, list_non_kf_indicies

    def stack_predictions(self, output_slow, output_combined, list_kf_indicies):
        t_dim_slow = output_slow.shape[1]
        t_dim_combined = output_combined.shape[1]
        batch_size = output_slow.shape[0]
        channels = output_slow.shape[2]
        height = output_slow.shape[3]
        width = output_slow.shape[4]
        T = t_dim_slow + t_dim_combined
        output = torch.zeros(batch_size, T, channels, height, width)
        for i in range(T):
            if i in list_kf_indicies:
                output[:, i, :, :, :] = output_slow[:, i, :, :, :]
            else:
                output[:, i, :, :, :] = output_combined[:, i, :, :, :]
        return output

    def run_encoder(self, input, encoder):
        # change the dimensions of the input tensor for the deeplabv3 encoder
        frame_batch_dim = input.shape[0]
        frames_t_dim = input.shape[1]
        frame_2d_batch = frame_batch_dim * frames_t_dim
        new_dim_frames = torch.reshape(input, (frame_2d_batch, input.shape[2], input.shape[3],
                                          input.shape[4]))
        encoder_output = encoder(new_dim_frames)
        return encoder_output, frame_batch_dim, frames_t_dim

    def reshape_output(self, input, batch_dim, t_dim):
        output = torch.reshape(input, (batch_dim, t_dim, input.shape[1],
                                      input.shape[2], input.shape[3]))
        return output

    # function to log the weights of the model
    def log_weights(self, step):
        # log the weights of the encoder
        writer.add_histogram("weights/MILADecoder/conv1/weight", model.conv1.weight.data, step)
        writer.add_histogram("weights/MILADecoder/conv1/bias", model.conv1.bias.data, step)
        writer.add_histogram("weights/MILADecoder/conv2/weight", model.conv2.weight.data, step)
        writer.add_histogram("weights/MILADecoder/conv2/bias", model.conv2.bias.data, step)
        writer.add_histogram("weights/MILADecoder/fc1/weight", model.fc1.weight.data, step)
        writer.add_histogram("weights/MILADecoder/fc1/bias", model.fc1.bias.data, step)
        writer.add_histogram("weights/MILADecoder/fc2/weight", model.fc2.weight.data, step)
        writer.add_histogram("weights/MILADecoder/fc2/bias", model.fc2.bias.data, step)


class SingleTaskModel(nn.Module):
    ''' This class contains the implementation of the slowfast model for a single task
            Args:
                enocder: A integer indicating the embedding size.
                decoder: A integer indicating the size of output dimension.
                number_frames: A integer indicating the hidden size of rnn.
                task: single task being
    '''

    def __init__(self, encoder_slow, encoder_fast, task_decoder, k=5, version='mila'):
        super().__init__()
        self.encoder_slow = encoder_slow
        self.encoder_fast = encoder_fast
        self.task_decoder = task_decoder
        # self.depth_decoder = depth_decoder
        self.K = k
        # self.atten = attention
        self.conv = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.version = version
        self.fast_se_feature = torch.zeros((8, 256, 8, 16))

    def forward(self, input):
        # [b, t, c, h, w]
        # Get the keyframes and non-keyframes for slow/fast setting
        keyframes, non_keyframes, list_kf_indicies = self.get_keyframes(input)

        # change the dimensions of the input tensor for the deeplabv3 encoder and obtain encoder features
        enc_slow_ftrs, batch_size, t_dim_slow = self.run_encoder(keyframes, self.encoder_slow)
        enc_fast_ftrs, _, t_dim_fast = self.run_encoder(non_keyframes, self.encoder_fast)

        # torch tile to make the slow tensor same dim as fast on dim=0
        enc_slow_ftrs_tiled = enc_slow_ftrs.tile((2, 1, 1, 1)) # is there a better way of doing this?

        # Different ways of propagating temporal features
        if self.version == 'basic':
            enc_combined_ftrs = torch.cat([enc_fast_ftrs, enc_slow_ftrs_tiled], dim=1)
            enc_slow_ftrs = self.conv(enc_slow_ftrs)
            seg_slow_pred = self.task_decoder(enc_slow_ftrs)
            seg_fast_pred = self.task_decoder(enc_combined_ftrs)
        else:
            seg_slow_pred, se_output_slow = self.task_decoder(enc_slow_ftrs, self.fast_se_feature)
            seg_fast_pred, se_output_fast = self.task_decoder(enc_fast_ftrs, se_output_slow)
            self.fast_se_feature = se_output_fast

        # Reshape the output tensors to a [B, T, C, H, W]
        output_seg_slow = self.reshape_output(seg_slow_pred, batch_size, t_dim_slow)
        output_seg_fast = self.reshape_output(seg_fast_pred, batch_size, t_dim_fast)

        # want to stake the predicted frames back in order of the T
        segmentation_pred = self.stack_predictions(output_seg_slow, output_seg_fast, list_kf_indicies)
        return segmentation_pred

    def get_keyframes(self, input):
        # input size dimension - [B, T, C, H, W]
        # create keyframes and non-keyframes using K
        T = input.shape[1]
        list_kf_indicies = [x for x in range(T) if x % self.K == 0]
        list_non_kf_indicies = list(set(range(T)) - set(list_kf_indicies))
        keyframes = input[:, list_kf_indicies, :, :, :]
        non_keyframes = input[:, list_non_kf_indicies, :, :, :]
        return keyframes, non_keyframes, list_kf_indicies, list_non_kf_indicies

    def stack_predictions(self, output_slow, output_combined, list_kf_indicies):
        t_dim_slow = output_slow.shape[1]
        t_dim_combined = output_combined.shape[1]
        batch_size = output_slow.shape[0]
        channels = output_slow.shape[2]
        height = output_slow.shape[3]
        width = output_slow.shape[4]
        T = t_dim_slow + t_dim_combined
        output = torch.zeros(batch_size, T, channels, height, width)
        for i in range(T):
            if i in list_kf_indicies:
                output[:, i, :, :, :] = output_slow[:, i, :, :, :]
            else:
                output[:, i, :, :, :] = output_combined[:, i, :, :, :]
        return output

    def run_encoder(self, input, encoder):
        # change the dimensions of the input tensor for the deeplabv3 encoder
        frame_batch_dim = input.shape[0]
        frames_t_dim = input.shape[1]
        frame_2d_batch = frame_batch_dim * frames_t_dim
        new_dim_frames = torch.reshape(input, (frame_2d_batch, input.shape[2], input.shape[3],
                                          input.shape[4]))
        encoder_output = encoder(new_dim_frames)
        return encoder_output, frame_batch_dim, frames_t_dim

    def reshape_output(self, input, batch_dim, t_dim):
        output = torch.reshape(input, (batch_dim, t_dim, input.shape[1],
                                      input.shape[2], input.shape[3]))
        return output

    # function to log the weights of the model
    def log_weights(self, step):
        # log the weights of the encoder
        writer.add_histogram("weights/MILADecoder/conv1/weight", model.conv1.weight.data, step)
        writer.add_histogram("weights/MILADecoder/conv1/bias", model.conv1.bias.data, step)
        writer.add_histogram("weights/MILADecoder/conv2/weight", model.conv2.weight.data, step)
        writer.add_histogram("weights/MILADecoder/conv2/bias", model.conv2.bias.data, step)
        writer.add_histogram("weights/MILADecoder/fc1/weight", model.fc1.weight.data, step)
        writer.add_histogram("weights/MILADecoder/fc1/bias", model.fc1.bias.data, step)
        writer.add_histogram("weights/MILADecoder/fc2/weight", model.fc2.weight.data, step)
        writer.add_histogram("weights/MILADecoder/fc2/bias", model.fc2.bias.data, step)



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
