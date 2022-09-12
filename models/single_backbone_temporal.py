import torch
import torch.nn as nn
from models.encoder import resnet50
# from mod import AblatedNet
from models.resnet_encoder import ResNet50, ResNet18
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
from models.deeplabv3_encoder import DeepLabv3
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
import pdb
from models.attention.attention import LocalContextAttentionBlock, GlobalContextAttentionBlock
from models.decoder import SegDecoder, DepthDecoder, MultiDecoder, DecoderTemporal, SegDecoder2, DepthDecoder2
from models.decoder import FeatureNoiseDecoder, FeatureDropDecoder, DropOutDecoder, CausalConv3d
from models.deeplabv3_encoder import DeepLabv3
from models.deeplabv3plus_encoder import DeepLabv3_plus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalModel2(nn.Module):
    ''' This class contains the implementation of the temporal slowfast setup (inspiration from original paper
    https://github.com/r1c7/SlowFastNetworks/blob/master/lib/slowfastnet.py )
            Args:
                encoder_slow: A integer indicating the embedding size.
                decode_fastr: A integer indicating the size of output dimension.
                task: which task is being done [segmentation, depth, depth_segmentation(multi-task)
                class_tasks: (int) if single class setup or list (int) of classes of depth and segmentation
                k: is the keyframe interval, and has to be greater than 1 and smaller than T (window_size)
                window_interval: is the number of frames we are processing as the T dimensions
                seg_drop_out: is the task decoder's dropout
                version: of the fusion concatenation or 3d temporal decoder
                semi-supervised: Adding semi supervision to the unlabelled data (segementation or depth) to the model
                multi_task: (bool) indicating whether task is a single or multi-task setup
    '''
    def __init__(self,
                 cfg,
                 task, class_tasks, seg_drop_out,
                 window_interval, k, semisup_loss,
                 unsup_loss,
                 version='basic',
                 mulit_task=False, causual_first_layer=True):
        super().__init__()
        _backbone_slow = cfg["model"]["backbone"]["encoder"]["resnet_slow"]
        # _backbone_fast = cfg["model"]["backbone"]["encoder"]["resnet_fast"]
        self.shared_encoder = DeepLabv3_plus(nInputChannels=3, output_dim=256, os=8, pretrained=True, _print=True)
        # self.encoder_slow = encoder_slow
        # self.encoder_fast = encoder_fast
        self.K = k
        self.version = version
        self.multi_task = mulit_task
        self.semi_sup = semisup_loss
        self.unsup = unsup_loss
        self.causal_conv = causual_first_layer
        input_dim_decoder = 304
        list_kf_indicies = [x for x in range(window_interval) if x % self.K == 0]
        list_non_kf_indicies = list(set(range(window_interval)) - set(list_kf_indicies))
        if version == 'sum_fusion':
            input_dim_decoder = 304
        if version == 'convnet_fusion':
            # number of keyframes + last_fast + annotated_frame
            input_dim_decoder = 304
            self.with_se_block = False
            # self.se_layer = SE_Layer(input_dim_decoder, 2)
            if self.causal_conv:
                self.convnet_fusion_layer = nn.Conv3d(input_dim_decoder, 304, kernel_size=(4, 1, 1), stride=1)
            if not self.causal_conv:
                self.convnet_fusion_layer = nn.Conv3d(input_dim_decoder, 304, kernel_size=(3, 1, 1), stride=1)

        # elif version == 'global_atten_fusion':
        #     input_dim_t = 256
        #     input_dim_decoder = 512
        #     output_dim = 256
        #     self.with_se_block = True
        #     self.se_layer = SE_Layer(input_dim_decoder, 2)
        #     self.global_attention_block = GlobalContextAttentionBlock(input_dim_t, output_dim, last_affine=True)
        if self.causal_conv:
            self.se_layer_slow = SE_Layer(304, 2)
            self.se_layer_fast = SE_Layer(304, 2)
            self.casual_conv_slow = nn.Sequential(CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3),),
                                                  CausalConv3d(in_channels=304 ,out_channels=304,kernel_size=(2, 3, 3), ), )
            self.casual_conv_fast = nn.Sequential(CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3), ),
                                                  CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3), ), )

        # Allocate the appropriate decoder for single task
        if task == 'segmentation':
            self.task_decoder = SegDecoder2(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm2d)
            if version == 'conv3d_fusion':
                input_dim_decoder = 304  # T dimension - but here its the number of keyframes + last fast frame
                self.task_decoder = DecoderTemporal(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm3d)
            # if version == 'global_atten_fusion':
            #     self.task_decoder = SegDecoder(512, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)

        if task == 'depth':
            self.task_decoder = DepthDecoder2(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)
            if version == 'conv3d_fusion':
                input_dim_decoder = 304  # T dimension - but here its the number of keyframes + last fast frame
                self.task_decoder = DecoderTemporal(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm3d)
            if version == 'global_atten_fusion':
                self.task_decoder = DepthDecoder(512, class_tasks, seg_drop_out, SynchronizedBatchNorm2d)

        # Set up the appropriate multi-task decoder for multi-task
        if task == 'depth_segmentation':
            depth_class = class_tasks[0]
            seg_class = class_tasks[1]
            input_dim_decoder = 304
            self.se_layer_depth = SE_Layer(304, 2)
            self.se_layer_seg = SE_Layer(304, 2)
            if self.causal_conv:
                self.casual_pass_depth = nn.Sequential(
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3), ),
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3), ), )
                self.casual_pass_seg = nn.Sequential(
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3), ),
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(2, 3, 3), ), )
            else:
                self.casual_pass_depth = nn.Sequential(
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(3, 3, 3), ),
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(3, 3, 3), ), )
                self.casual_pass_seg = nn.Sequential(
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(3, 3, 3), ),
                    CausalConv3d(in_channels=304, out_channels=304, kernel_size=(3, 3, 3), ), )
            # self.task_decoder = MultiDecoder(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)
            depth_dec = [DepthDecoder2(input_dim_decoder, depth_class, seg_drop_out, SynchronizedBatchNorm2d)]
            seg_dec = [SegDecoder2(input_dim_decoder, seg_class, seg_drop_out, SynchronizedBatchNorm2d)]
            self.list_decoders = nn.ModuleList([*depth_dec, *seg_dec])

            if version == 'conv3d_fusion':
                input_dim_decoder = 304  # T dimension - but here its the number of keyframes + last fast frame
                depth_dec = [DecoderTemporal(input_dim_decoder, depth_class, seg_drop_out, SynchronizedBatchNorm2d)]
                seg_dec = [DecoderTemporal(input_dim_decoder, seg_class, seg_drop_out, SynchronizedBatchNorm2d)]
                self.list_decoders = nn.ModuleList([*depth_dec, *seg_dec])


    def forward(self, input):
        # [b, t, c, h, w]
        # print(input[:, -1])

        #print(input.shape) # torch.Size([4, 3, 3, 128, 256])
        enc_ftrs, batch_size, t_dim = self.run_encoder(input, self.shared_encoder)
        print(enc_ftrs.shape) #- torch.Size([4, 3, 304, 128, 256])
        output_ftrs = self.reshape_output(enc_ftrs, batch_size, t_dim)
        print(output_ftrs.shape)
        task_predictions = 0
        if not self.multi_task:
            # Different ways of propagating temporal features
            if self.version == 'sum_fusion':
                task_predictions = self.sum_fusion(output_ftrs, mulit_task=self.multi_task, causal=self.causal_conv)
            elif self.version == 'conv3d_fusion':
                task_predictions = self.temporal_net(output_ftrs, mulit_task=self.multi_task, causal=self.causal_conv)
            elif self.version == 'convnet_fusion':
                task_predictions = self.convnet_fusion(output_ftrs, with_se_block=False, causal=self.causal_conv)

        if self.multi_task:
            if self.version == 'sum_fusion':
                task_predictions = self.sum_fusion(output_ftrs, mulit_task=self.multi_task, causal=self.causal_conv)
            elif self.version == 'conv3d_fusion':
                task_predictions = self.temporal_net(output_ftrs, mulit_task=self.multi_task, causal=self.causal_conv)
            elif self.version == 'convnet_fusion':
                task_predictions = self.convnet_fusion(output_ftrs, with_se_block=False, causal=self.causal_conv)
            elif self.version == 'causal_fusion':
                task_predictions = self.causal_module(output_ftrs, causal=self.causal_conv)

        # Output based on model choice
        return {'supervised': task_predictions}

    def add_casaul_module(self, output_ftrs):
        slow_output = torch.stack([self.se_layer_slow(output_ftrs[:, 0]),
                                   self.se_layer_slow(output_ftrs[:, 2])], dim=1)
        fast_output = torch.stack([self.se_layer_fast(output_ftrs[:, 1]),
                                   self.se_layer_fast(output_ftrs[:, 2])], dim=1)
        x_fusion_input_slow = slow_output.permute(0, 2, 1, 3, 4)
        x_fusion_input_fast = fast_output.permute(0, 2, 1, 3, 4)
        x_fusion_causal_slow = self.casual_conv_slow(x_fusion_input_slow)
        x_fusion_causal_fast = self.casual_conv_fast(x_fusion_input_fast)
        x_fusion_causal_slow = x_fusion_causal_slow.permute(0, 2, 1, 3, 4)
        x_fusion_causal_fast = x_fusion_causal_fast.permute(0, 2, 1, 3, 4)
        x_fusion_input = torch.cat([x_fusion_causal_slow, x_fusion_causal_fast], dim=1)
        return x_fusion_input

    def sum_fusion(self, enc_ftrs, mulit_task=False, causal=False):
        # x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        if not mulit_task:
            if causal:
                enc_ftrs = self.add_casaul_module(enc_ftrs)

            x_fusion = torch.sum(enc_ftrs, dim=1).squeeze(1)
            task_predictions = self.task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
        else:
            # pass the average fusion to segmentation but not depth
            if causal:
                enc_ftrs = self.add_casaul_module(enc_ftrs)
                x_fusion_input_seg = torch.stack([self.se_layer_seg(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                                 dim=1)
                x_fusion_input_depth = torch.stack([self.se_layer_depth(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                                   dim=1)
            else:
                x_fusion_input_seg = torch.stack([self.se_layer_seg(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                                 dim=1)
                x_fusion_input_depth = torch.stack([self.se_layer_depth(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                                   dim=1)
            x_fusion_sum_seg = torch.sum(x_fusion_input_seg, dim=1)
            x_fusion_sum_depth = torch.sum(x_fusion_input_depth, dim=1)
            depth_pred = self.list_decoders[0](x_fusion_sum_depth)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](x_fusion_sum_seg)
            seg_pred = seg_pred.squeeze(1)
            task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def temporal_net(self, enc_ftrs, mulit_task=False, causal=False):
        # x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        # x_fusion = torch.cat([output_slow, fast_frame], dim=1)
        if not mulit_task:
            if causal:
                enc_ftrs = self.add_casaul_module(enc_ftrs)
                # task_predictions = self.task_decoder(enc_ftrs)
                # task_predictions = task_predictions.squeeze(1)
            task_predictions = self.task_decoder(enc_ftrs)
            task_predictions = task_predictions.squeeze(1)
        else:
            if causal:
                enc_ftrs = self.add_casaul_module(enc_ftrs)

            x_fusion_input_seg = torch.stack([self.se_layer_seg(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                             dim=1)
            x_fusion_input_depth = torch.stack([self.se_layer_depth(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                               dim=1)
            x_fusion_input_depth = x_fusion_input_depth.permute(0, 2, 1, 3, 4)
            causal_depth = self.casual_pass_depth(x_fusion_input_depth)
            causal_depth = causal_depth.permute(0, 2, 1, 3, 4)

            x_fusion_input_seg = x_fusion_input_seg.permute(0, 2, 1, 3, 4)
            causal_seg = self.casual_pass_depth(x_fusion_input_seg)
            causal_seg = causal_seg.permute(0, 2, 1, 3, 4)

            depth_pred = self.list_decoders[0](causal_depth)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](causal_seg)
            seg_pred = seg_pred.squeeze(1)
            task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def causal_module(self, enc_ftrs, causal=False):
        # x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        # pass through SE layer
        if causal:
            enc_ftrs = self.add_casaul_module(enc_ftrs)

        x_fusion_input_seg = torch.stack([self.se_layer_seg(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                         dim=1)
        x_fusion_input_depth = torch.stack([self.se_layer_depth(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                           dim=1)
        # Through the casual module
        x_fusion_input_depth = x_fusion_input_depth.permute(0, 2, 1, 3, 4)
        x_fusion_causal_depth = self.casual_pass_depth(x_fusion_input_depth)[:, :, -1, ]
        x_fusion_input_seg = x_fusion_input_seg.permute(0, 2, 1, 3, 4)
        x_fusion_causal_seg = self.casual_pass_seg(x_fusion_input_seg)[:, :, -1, ]

        # run through task specific decoders
        depth_pred = self.list_decoders[0](x_fusion_causal_depth)
        depth_pred = depth_pred.squeeze(1)
        seg_pred = self.list_decoders[1](x_fusion_causal_seg)
        seg_pred = seg_pred.squeeze(1)
        task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def convnet_fusion(self, enc_ftrs, with_se_block=False, causal=False):
        # take the slow output of the keyframes and the last frame of the fast
        # x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        if causal:
            enc_ftrs = self.add_casaul_module(enc_ftrs)

        if not self.multi_task:
            x_fusion = enc_ftrs.permute(0, 2, 1, 3, 4)
            print(x_fusion.shape)
            x_fusion = self.convnet_fusion_layer(x_fusion)
            x_fusion = x_fusion.permute(0, 1, 2, 3, 4).squeeze()
            task_predictions = self.task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
        else:

            x_fusion_input_seg = torch.stack([self.se_layer_seg(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                             dim=1)
            x_fusion_input_depth = torch.stack([self.se_layer_depth(enc_ftrs[:, i]) for i in range(enc_ftrs.shape[1])],
                                               dim=1)

            x_fusion_seg = x_fusion_input_seg.permute(0, 2, 1, 3, 4)
            x_fusion_seg = self.convnet_fusion_layer(x_fusion_seg)
            x_fusion_seg = x_fusion_seg.permute(0, 1, 2, 3, 4).squeeze()

            x_fusion_depth = x_fusion_input_depth.permute(0, 2, 1, 3, 4)
            x_fusion_depth = self.convnet_fusion_layer(x_fusion_depth)
            x_fusion_depth = x_fusion_depth.permute(0, 1, 2, 3, 4).squeeze()

            depth_pred = self.list_decoders[0](x_fusion_depth)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](x_fusion_seg)
            seg_pred = seg_pred.squeeze(1)
            task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def global_attention_fusion(self, output_slow, output_fast, list_kf_indicies, list_non_kf_indicies, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        # for loop over the output slow tensor, doing a query to the annotated frame then
        attention_fusion = []
        if max(list_kf_indicies) > max(list_non_kf_indicies):
            if len(list_kf_indicies) > 1:
                annotated_frame = output_slow[:, -1]  # [2, 256, 8, 16]
                output_slow = output_slow[:, :-1]
                attention_fusion = [self.global_attention_block(output_slow[:, i], annotated_frame)
                                    for i in range(output_slow.shape[1])]
                attention_fusion.append(self.global_attention_block(output_fast[:, -1], annotated_frame))
            else:
                # last frame is the only keyframe
                annotated_frame = output_slow[:, -1]
                attention_fusion = [self.global_attention_block(output_fast[:, -1], output_slow[:, -1])]
        else:
            if len(list_non_kf_indicies) > 1:
                annotated_frame = output_fast[:, -1]
                output_fast = output_fast[:, :-1]
                attention_fusion = [self.global_attention_block(output_slow[:, i], annotated_frame)
                                    for i in range(output_slow.shape[1])]
                attention_fusion.append(self.global_attention_block(output_fast[:, -1], annotated_frame))
            else:
                # last frame is a fast_frame and there is no other fast frames, could be many keyframes
                annotated_frame = output_fast[:, -1]
                attention_fusion = [self.global_attention_block(output_slow[:, i], output_fast[:, -1])
                                    for i in range(output_slow.shape[1])]

        # stack the outputs of the global attention tensors
        attn_fusion_tensor = torch.cat([x for x in attention_fusion], dim=1) # [4, 512, 8, 16]
        # print('before se layer')
        # print(attn_fusion_tensor.shape)
        # print(fast_frame.shape) - [4, 256, 8, 16]
        # print(kf_encoded_frames.shape) - [4, 256, 8, 16]
        if self.with_se_block:
            # x_fusion = attn_fusion_tensor.unsqueeze(1)
            attn_fusion_tensor = self.se_layer(attn_fusion_tensor)
        if not mulit_task:
            #  depth decoder or segmentation decoder
            task_predictions = self.task_decoder(attn_fusion_tensor)
            task_predictions = task_predictions.squeeze(1)
        else:
            # pass the average fusion to segmentation but not depth
            depth_pred = self.list_decoders[0](annotated_frame)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](attn_fusion_tensor)
            seg_pred = seg_pred.squeeze(1)
            task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def reshape_input(self, tensor_input):
        frame_batch_dim = tensor_input.shape[0]
        frames_t_dim = tensor_input.shape[1]
        frame_2d_batch = frame_batch_dim * frames_t_dim
        tensor_input = torch.reshape(tensor_input, (frame_2d_batch, tensor_input.shape[2],
                                                                         tensor_input.shape[3],
                                                                         tensor_input.shape[4]))
        return tensor_input, frame_batch_dim, frames_t_dim

    #  labelled_pred, unlabelled_preds = self.run_perturbed_decoders(all_encoded_frames)
    def run_perturbed_decoders(self, all_encoded_frames, index_of_labelled_frame):
        # Reshape all the labelled frames
        all_encoded_frames_reshaped, all_frame_batch, all_frame_t = self.reshape_input(all_encoded_frames)
        list_of_unlabelled_frames = [x for x in range(all_frame_t) if x != index_of_labelled_frame]
        unlabelled_encoded_frames = all_encoded_frames[:, list_of_unlabelled_frames]
        unlabelled_encoded_frames, frame_batch, unlabelled_t_dim = self.reshape_input(unlabelled_encoded_frames)

        # run through main decoder
        output_main = self.task_decoder(all_encoded_frames_reshaped)
        output_main = self.reshape_output(output_main, all_frame_batch, all_frame_t)
        main_unlabelled_pred = output_main[:, list_of_unlabelled_frames]
        main_labelled_pred = output_main[:, index_of_labelled_frame]
        # get unlabelled frames
        # task predictions for each set or unlabelled frames
        perturbed_unlabelled_pred = []
        for decoder in self.aux_decoder:
            # send the unlabelled frames through each perturbed decoders to create 3 sets of
            output = decoder(unlabelled_encoded_frames)
            output = output.squeeze(1)
            output = self.reshape_output(output, frame_batch, unlabelled_t_dim)
            perturbed_unlabelled_pred.append(output)
        return main_labelled_pred, main_unlabelled_pred, perturbed_unlabelled_pred

    def get_keyframes(self, input):
        # input size dimension - [B, T, C, H, W]
        # create keyframes and non-keyframes using K
        T = input.shape[1]
        list_kf_indicies = [x for x in range(T) if x % self.K == 0]
        list_non_kf_indicies = list(set(range(T)) - set(list_kf_indicies))
        keyframes = input[:, list_kf_indicies, :, :, :]
        non_keyframes = input[:, list_non_kf_indicies, :, :, :]
        return keyframes, non_keyframes, list_kf_indicies, list_non_kf_indicies

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
        # print('have to reshape tensor for non-temporal fit to deeplabv3: '+str(new_dim_frames.shape))
        encoder_output = encoder(new_dim_frames)
        return encoder_output, frame_batch_dim, frames_t_dim

    def reshape_output(self, input_, batch_dim, t_dim):
        output = torch.reshape(input_, (batch_dim, t_dim, input_.shape[1],
                                        input_.shape[2], input_.shape[3]))
        return output


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
