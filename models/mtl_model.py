import torch
import torch.nn as nn
from models.decoder import Decoder, MILADecoder, SegDecoder
from models.encoder import resnet50
# from mod import AblatedNet
from models.resnet_encoder import ResNet50, ResNet18
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
from models.deeplabv3_encoder import DeepLabv3
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, SynchronizedBatchNorm2d
import pdb
from models.attention.attention import LocalContextAttentionBlock, GlobalContextAttentionBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class MultiTaskModel1(nn.Module):
    ''' This class contains the implementation of the slowfast model.
    https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py
            Args:
                enocder: A integer indicating the embedding size.
                decoder: A integer indicating the size of output dimension.
                number_frames: A integer indicating the hidden size of rnn.
    '''

    def __init__(self, encoder_slow, encoder_fast, list_decoders, k=2, version='basic', mulit_task=False):
        super().__init__()
        self.encoder_slow = encoder_slow
        self.encoder_fast = encoder_fast
        self.list_decoders = list_decoders
        self.K = k
        self.conv = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.version = version
        self.multi_task = mulit_task
        self.fast_se_feature = torch.zeros((8, 256, 8, 16))
        if version == 'globalAtten':
            input_dim_t = 256
            output_dim = 256
            self.global_attention_block = GlobalContextAttentionBlock(input_dim_t, output_dim, last_affine=True).to('cuda:0')

    def forward(self, input):
        # [b, t, c, h, w]
        # Get the keyframes and non-keyframes for slow/fast setting
        # keyframes, non_keyframes, list_kf_indicies = self.get_keyframes(input)
        # print('input shape: ' +str(input.shape)) #- torch.Size([4, 5, 3, 128, 256])
        # pdb.set_trace()
        # keyframes, non_keyframes, kf_mask, nk_mask = self.keyframes(input)
        keyframes, non_keyframes, list_kf_indicies, list_non_kf_indicies = self.get_keyframes(input)
        # print('keyframe inputs using masking technique, [image, 0, image, 0, image(annotated)]: ' + str(keyframes.shape))
        # print('non-keyframe inputs using masking technique, [0, image, 0, image, 0]: ' + str(non_keyframes.shape))
        # change the dimensions of the input tensor for the deeplabv3 encoder and obtain encoder features
        enc_slow_ftrs, batch_size, t_dim_slow = self.run_encoder(keyframes, self.encoder_slow)
        enc_fast_ftrs, _, t_dim_fast = self.run_encoder(non_keyframes, self.encoder_fast)
        output_slow = self.reshape_output(enc_slow_ftrs, batch_size, t_dim_slow)
        output_fast = self.reshape_output(enc_fast_ftrs, batch_size, t_dim_fast)
        # print('Reshaped output from slow (b,t,c,w,h):' + str(output_slow.shape))
        # print('Reshaped output from fast (b,t,c,w,h):' + str(output_fast.shape))
        # print('enc_slow_ftrs: ' + str(output_slow.shape))
        # print('enc_fast_ftrs[:,-1]: ' + str(torch.unsqueeze(output_fast[:, -1], dim=1).shape))
        # Different ways of propagating temporal features
        if self.version == 'basic':
            task_predictions = self.average(output_slow, output_fast, mulit_task=self.multi_task)
        elif self.version == 'advers':
            task_predictions = self.adverserial_concatentation(input, mulit_task=self.multi_task)
        elif self.version == 'temporal_network':
            task_predictions = self.temporal_net(output_slow, output_fast, mulit_task=self.multi_task)
        elif self.version == 'convnet_fusion':
            task_predictions = self.convnet_fusion(output_slow, output_fast, with_se_block=False, mulit_task=self.multi_task)
        elif self.version == 'globalAtten':
            task_predictions = self.global_attention_fusion(output_slow, output_fast, mulit_task=self.multi_task)
        elif self.version == 'localAtten':
            pass
        else:
            pass

        # Reshape the output tensors to a [B, T, C, H, W]
        # output_seg_slow = self.reshape_output(seg_slow_pred, batch_size, t_dim_slow)
        # output_seg_fast = self.reshape_output(seg_fast_pred, batch_size, t_dim_fast)

        # want to stake the predicted frames back in order of the T
        # segmentation_pred = self.stack_predictions(output_seg_slow, output_seg_fast, list_kf_indicies)
        # print('task_predictions at the end of mtl_model:' + str(task_predictions.shape))
        return task_predictions

    def average(self, output_slow, output_fast, mulit_task=False):
        # frame_batch_dim = output_slow.shape[0]
        # print(frame_batch_dim)
        # frames_t_dim = output_slow.shape[1]
        # frame_2d_batch = frame_batch_dim * frames_t_dim
        # kf_encoded_frames = torch.reshape(output_slow, (frame_2d_batch, output_slow.shape[2], output_slow.shape[3],
        #                                                 output_slow.shape[4])).to('cuda:0')

        # fast_frame = output_fast[:, -1].tile((2, 1, 1, 1)).to('cuda:0')  # is there a better way of doing this?
        # print(kf_encoded_frames.shape)# - torch.Size([8, 256, 8, 16])
        # print(fast_frame.shape) #- torch.Size([8, 256, 8, 16])
        # output_fast_unsqueezed = torch.unsqueeze(output_fast[:, -1], dim=1)
        # print('kf_encoded_frames: ' + str(kf_encoded_frames.shape))
        # print('fast_frame: ' + str(fast_frame.shape))
        # x_fusion = torch.cat([kf_encoded_frames, fast_frame], dim=1).to('cuda:0')
        output_slow = output_slow.to('cuda:0')
        fast_frame = output_fast[:, -1].unsqueeze(1).to('cuda:0')
        x_fusion = torch.cat([output_slow, fast_frame], dim=1).to('cuda:0')
        x_fusion = torch.mean(x_fusion, dim=1).squeeze(1)
        if not mulit_task:
            #  depth decoder or segmentation decoder
            task_decoder = self.list_decoders[-1]
            task_predictions = task_decoder(x_fusion)
        else:
            task_predictions = []
            for task_decoder in self.list_decoders:
                # Reshape the output tensors to a [B, T, C, H, W]
                # output_slow = self.reshape_output(task_decoder(enc_slow_ftrs), batch_size, t_dim_slow)
                # output_fast = self.reshape_output(task_decoder(enc_combined_ftrs), batch_size, t_dim_fast)
                task_predictions.append(task_decoder(x_fusion))
        return task_predictions

    def adverserial_concatentation(self, input, mulit_task=False):
        keyframes, non_keyframes, list_kf_indicies, list_non_kf_indicies = self.get_keyframes(input)
        # print(input.shape) - torch.Size([4, 5, 3, 128, 256])
        x_slow_all, batch_size, t_dim_slow = self.run_encoder(input, self.encoder_slow)
        x_fast_all, _, t_dim_fast = self.run_encoder(input, self.encoder_fast)
        # print(x_fast_all.shape) - torch.Size([10, 256, 8, 16])
        # print(x_slow_all.shape) - torch.Size([10, 2, 8, 16])
        # apply masks to x_slow_all and x_fast_all and put representations into correct keyframe and frames
        # only take the last frame before segmentation + the first keyframe
        # print(x_slow_all[:, list_kf_indicies].shape)
        # print(x_fast_all[:, list_non_kf_indicies][-1].shape)
        fast_last_frame = x_fast_all[:, list_non_kf_indicies][-1].tile((x_slow_all[:, list_kf_indicies].shape[0], 1, 1, 1))
        x_fused = torch.cat([x_slow_all[:, list_kf_indicies], fast_last_frame], dim=1).to('cuda:0')
        if not mulit_task:
            task_decoder = self.list_decoders[-1]
            task_predictions = task_decoder(x_fused)
        # x_fusion = self.fusion_network(x_fused)
        # y = self.decoder(x_fusion)
        # return y, x_slow_all, x_fast_all
        # output_fast_unsqueezed = torch.unsqueeze(output_fast[:, -1], dim=1)
        return task_predictions

    def temporal_net(self, output_slow, output_fast, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        output_slow = output_slow.to('cuda:0')
        fast_frame = output_fast[:, -1].unsqueeze(1).to('cuda:0')
        x_fusion = torch.cat([output_slow, fast_frame], dim=1).to('cuda:0')
        if not mulit_task:
            #  depth decoder or segmentation decoder
            task_decoder = self.list_decoders[-1]
            task_predictions = task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
        else:
            task_predictions = []
            for task_decoder in self.list_decoders:
                task_pred = task_decoder(x_fusion)
                task_pred = task_pred.squeeze(1)
                task_predictions.append(task_pred)
        return task_predictions

    def convnet_fusion(self, output_slow, output_fast, with_se_block=False, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        output_slow = output_slow.to('cuda:0')
        fast_frame = output_fast[:, -1].unsqueeze(1).to('cuda:0')
        x_fusion = torch.cat([output_slow, fast_frame], dim=1).to('cuda:0')
        conv_layer = nn.Conv3d(x_fusion.shape[1], 1, kernel_size=(1, 1, 1), stride=1).to('cuda:0')
        x_fusion = conv_layer(x_fusion).squeeze()
        if with_se_block:
            x_fusion = x_fusion.unsqueeze(1)
            se_block = ChannelSELayer3D(x_fusion.shape[1], 2).to('cuda:0')
            x_fusion = se_block(x_fusion)

        if not mulit_task:
            #  depth decoder or segmentation decoder
            task_decoder = self.list_decoders[-1]
            task_predictions = task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
        else:
            task_predictions = []
            for task_decoder in self.list_decoders:
                task_pred = task_decoder(x_fusion)
                task_pred = task_pred.squeeze(1)
                task_predictions.append(task_pred)
        return task_predictions

    def global_attention_fusion(self, output_slow, output_fast, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        # output_slow = output_slow.to('cuda:0')
        frame_batch_dim = output_slow.shape[0]
        frames_t_dim = output_slow.shape[1]
        frame_2d_batch = frame_batch_dim * frames_t_dim
        kf_encoded_frames = torch.reshape(output_slow, (frame_2d_batch, output_slow.shape[2], output_slow.shape[3],
                                                        output_slow.shape[4])).to('cuda:0')
        # print(kf_encoded_frames.shape)
        fast_frame = output_fast[:, -1]
        # print(fast_frame.shape)
        # fast_frame = fast_frame.tile((2,)).to('cuda:0')
        fast_frame = fast_frame.repeat(2, 1, 1, 1).to('cuda:0')
        # print(fast_frame.shape) - [4, 256, 8, 16]
        # print(kf_encoded_frames.shape) - [4, 256, 8, 16]
        # x_fusion = torch.cat([output_slow, fast_frame], dim=1).to('cuda:0')
        x_fusion = self.global_attention_block(kf_encoded_frames, fast_frame)
        print('After fusion attention: ' + str(x_fusion.shape))
        # conv_layer = nn.Conv3d(x_fusion.shape[1], 1, kernel_size=(1, 1, 1), stride=1).to('cuda:0')
        # x_fusion = conv_layer(x_fusion).squeeze()

        # if with_se_block:
        #     x_fusion = x_fusion.unsqueeze(1)
        #     se_block = ChannelSELayer3D(x_fusion.shape[1], 2).to('cuda:0')
        #     x_fusion = se_block(x_fusion)

        if not mulit_task:
            #  depth decoder or segmentation decoder
            task_decoder = self.list_decoders[-1]
            task_predictions = task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
            print('task_predictions.shape: ' + str(task_predictions.shape))
        else:
            task_predictions = []
            for task_decoder in self.list_decoders:
                task_pred = task_decoder(x_fusion)
                task_pred = task_pred.squeeze(1)
                task_predictions.append(task_pred)
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

    # def get_keyframes(self, input):
    #     # input size dimension - [B, T, C, H, W]
    #     # create keyframes and non-keyframes using K
    #     T = input.shape[1]
    #     list_kf_indicies = [x for x in range(T) if x % self.K == 0]
    #     list_non_kf_indicies = list(set(range(T)) - set(list_kf_indicies))
    #     keyframes = input[:, list_kf_indicies, :, :, :]
    #     non_keyframes = input[:, list_non_kf_indicies, :, :, :]
    #     return keyframes, non_keyframes

    def keyframes(self, input):
        T = input.shape[1]
        kf_mask = [1 if index % self.K == 0 else 0 for index in range(T)]
        nk_mask = [1 if index % self.K != 0 else 0 for index in range(T)]
        return input[:, kf_mask], input[:, nk_mask], kf_mask, nk_mask

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
        print('have to reshape tensor for non-temporal fit to deeplabv3: '+str(new_dim_frames.shape))
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
        print(x.shape)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        print(y.shape)
        return x * y.expand_as(x)
class ChannelSELayer3D(nn.Module):
    # https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, D, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class SimpleNet(nn.Module):
    ''' SimpleNet is a network that takes in two tensors of features, one from the slow encoded features
    and the other the fast features frames and outputs a supposedly better performing concatenation of the features'''
    def __init__(self, input_dim_slow, input_dim_fast, output_dim):
        super(SimpleNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv_layer = nn.Sequential(
                nn.Conv3d(mid_input_dim, mid_input_dim, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
                BatchNorm(mid_input_dim),
                nn.ReLU(),
                nn.Dropout(drop_out),
                nn.Conv3d(mid_input_dim, 1, kernel_size=(1, 1, 1), stride=1),
        )
        self._init_weight()

    def forward(self, output_slow, output_fast):
        frame_batch_dim = output_slow.shape[0]
        print(frame_batch_dim)
        frames_t_dim = output_slow.shape[1]
        frame_2d_batch = frame_batch_dim * frames_t_dim
        kf_encoded_frames = torch.reshape(output_slow, (frame_2d_batch, output_slow.shape[2], output_slow.shape[3],
                                                        output_slow.shape[4]))

        fast_frame = output_fast[:, -1].tile((2, 1, 1, 1))
        combined = torch.cat((c.view(c.size(0), -1),
                              f.view(f.size(0), -1)), dim=1)
        # then pass both through a convolutional network

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # output the concatenated features of

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


if __name__ == "__main__":
    num_classes = 19
    prev_sblock_kf = 0
    L = 5
    INPUT_DIM = 512
    image_size = 256
    CLASS_TASKS = 19
    drop_out = 0.5
    # dec_seg = MILADecoder(input_dim=256)
    dec = SegDecoder(INPUT_DIM, CLASS_TASKS, drop_out, SynchronizedBatchNorm1d).to(device)
    # dec_seg = Decoder(input_dim=INPUT_DIM)
    # net = AblatedNet(c_in=3, c_out_seg=3)
    # encoder_slow = ResNet50(19, 3)
    # encoder_slow = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    # encoder_slow = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=19)
    # encoder_fast = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    encoder_slow = DeepLabv3('resnet50').to('cpu')
    encoder_fast = DeepLabv3('resnet18').to('cpu')
    # encoder_fast = ResNet18(19, 3)
    batch = 4
    T = 5
    channels = 3
    height = 128
    width = 256
    input_tensor = torch.autograd.Variable(torch.rand(batch, T, channels, height, width)).to(device)
    # semantic_pred = net(input_tensor)
    model = MultiTaskModel1(encoder_slow, encoder_fast, [dec], 4, version='basic').to(device)
    semantic_pred = model(input_tensor)
    print(semantic_pred.shape)
