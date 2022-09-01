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
from models.decoder import SegDecoder, DepthDecoder, MultiDecoder, DecoderTemporal
from models.decoder import FeatureNoiseDecoder, FeatureDropDecoder, DropOutDecoder
from models.deeplabv3_encoder import DeepLabv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemporalModel(nn.Module):
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
                 mulit_task=False):
        super().__init__()
        _backbone_slow = cfg["model"]["backbone"]["encoder"]["resnet_slow"]
        _backbone_fast = cfg["model"]["backbone"]["encoder"]["resnet_fast"]
        self.encoder_slow = DeepLabv3(_backbone_slow)
        self.encoder_fast = DeepLabv3(_backbone_fast)
        # self.encoder_slow = encoder_slow
        # self.encoder_fast = encoder_fast
        self.K = k
        self.version = version
        self.multi_task = mulit_task
        self.semi_sup = semisup_loss
        self.unsup = unsup_loss
        input_dim_decoder = 256
        list_kf_indicies = [x for x in range(window_interval) if x % self.K == 0]
        list_non_kf_indicies = list(set(range(window_interval)) - set(list_kf_indicies))
        if version == 'average_fusion':
            input_dim_decoder = 256
        elif version == 'advers':
            input_dim_decoder = 5
        elif version == 'convnet_fusion':
            # number of keyframes + last_fast + annotated_frame
            if len(list_kf_indicies) > len(list_non_kf_indicies):
                if len(list_kf_indicies) > 1:
                    t_dim = len(list_kf_indicies) + 1
                else:
                    t_dim = 2
            else:
                if len(list_non_kf_indicies) > 1:
                    t_dim = len(list_kf_indicies) + 2
                else:
                    t_dim = 2
            input_dim_decoder = 256
            self.with_se_block = True
            self.se_layer = SE_Layer(input_dim_decoder, 2)
            self.convnet_fusion_layer = nn.Conv3d(input_dim_decoder, 256, kernel_size=(t_dim, 1, 1), stride=1)
        elif version == 'global_atten_fusion':
            input_dim_t = 256
            input_dim_decoder = 512
            output_dim = 256
            self.with_se_block = True
            self.se_layer = SE_Layer(input_dim_decoder, 2)
            self.global_attention_block = GlobalContextAttentionBlock(input_dim_t, output_dim, last_affine=True)
        # elif version == 'local_atten_fusion':
        #     input_dim_t = 256
        #     output_dim = 256
        #     kernel_neighbourhood = 9
        #     self.self.local_attention_block = LocalContextAttentionBlock(input_dim_t, output_dim,
        #                                                                  kernel_neighbourhood, last_affine=True)

        if self.semi_sup:
            # instantiate the different pertubation decoders for the unlabelled data
            input_dim_decoder = 256
            input_dim = 256
            if len(list_kf_indicies) > len(list_non_kf_indicies):
                self.index_of_labelled_frame = len(list_kf_indicies) - 1
            else:
                self.index_of_labelled_frame = len(list_kf_indicies) + len(list_non_kf_indicies) - 1
            # feature_drop = [FeatureDropDecoder(input_dim, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)]
            feature_noise = [FeatureNoiseDecoder(input_dim, class_tasks, seg_drop_out, SynchronizedBatchNorm1d, uniform_range=0.3)]
            drop_decoder = [DropOutDecoder(input_dim, class_tasks, seg_drop_out, SynchronizedBatchNorm1d, drop_rate=0.3,
                                          spatial_dropout=True)]
            # self.aux_decoder = nn.ModuleList([*drop_decoder, *feature_noise, *feature_drop])
            self.aux_decoder = nn.ModuleList([*drop_decoder, *feature_noise])
            # self.convnet_fusion_layer2 = nn.Conv3d(input_dim_decoder, 256, kernel_size=(unlabelled_num_frames, 1, 1),
            #                                        stride=1)
            # self.se_layer_2 = SE_Layer(input_dim_decoder, 2)

        # Allocate the appropriate decoder for single task
        if task == 'segmentation':
            self.task_decoder = SegDecoder(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)
            if version == 'conv3d_fusion':
                input_dim_decoder = 256  # T dimension - but here its the number of keyframes + last fast frame
                self.task_decoder = DecoderTemporal(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm3d)
            if version == 'global_atten_fusion':
                self.task_decoder = SegDecoder(512, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)

        if task == 'depth':
            self.task_decoder = DepthDecoder(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)
            if version == 'conv3d_fusion':
                input_dim_decoder = 256  # T dimension - but here its the number of keyframes + last fast frame
                self.task_decoder = DecoderTemporal(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm3d)
            if version == 'global_atten_fusion':
                self.task_decoder = DepthDecoder(512, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)
        # Set up the appropriate multi-task decoder for multi-task
        elif task == 'depth_segmentation':
            depth_class = class_tasks[0]
            seg_class = class_tasks[1]
            # self.task_decoder = MultiDecoder(input_dim_decoder, class_tasks, seg_drop_out, SynchronizedBatchNorm1d)
            depth_dec = [DepthDecoder(input_dim_decoder, depth_class, seg_drop_out, SynchronizedBatchNorm1d)]
            seg_depth = [SegDecoder(input_dim_decoder, seg_class, seg_drop_out, SynchronizedBatchNorm1d)]
            self.list_decoders = nn.ModuleList([*depth_dec, *seg_depth])

            if version == 'conv3d_fusion':
                input_dim_decoder = 256  # T dimension - but here its the number of keyframes + last fast frame
                depth_dec = [DecoderTemporal(input_dim_decoder, depth_class, seg_drop_out, SynchronizedBatchNorm1d)]
                seg_depth = [DecoderTemporal(input_dim_decoder, seg_class, seg_drop_out, SynchronizedBatchNorm1d)]
                self.list_decoders = nn.ModuleList([*depth_dec, *seg_depth])


    def forward(self, input):
        # [b, t, c, h, w]
        # Get the keyframes and non-keyframes for slow/fast setting
        print('input shape: ' + str(input.shape))  # - torch.Size([4, 5, 3, 128, 256])
        keyframes, non_keyframes, list_kf_indicies, list_non_kf_indicies = self.get_keyframes(input)

        # check if we are to using regularise the encoders then need the output of all frames in encoders
        if self.unsup['Mode']:
            x_slow_all, batch_size, t_dim_slow = self.run_encoder(input, self.encoder_slow)
            x_fast_all, _, t_dim_fast = self.run_encoder(input, self.encoder_fast) # [12, 256, 8, 16]
            x_slow_all = self.reshape_output(x_slow_all, input.shape[0], input.shape[1])
            x_fast_all = self.reshape_output(x_fast_all, input.shape[0], input.shape[1])
            output_slow = x_slow_all[:, list_kf_indicies]
            output_fast = x_fast_all[:, list_non_kf_indicies]

        if not self.unsup['Mode']:
            enc_fast_ftrs, _, t_dim_fast = self.run_encoder(non_keyframes, self.encoder_fast)
            enc_slow_ftrs, batch_size, t_dim_slow = self.run_encoder(keyframes, self.encoder_slow) # [8, 256, 8, 16]
            output_slow = self.reshape_output(enc_slow_ftrs, batch_size, t_dim_slow)
            output_fast = self.reshape_output(enc_fast_ftrs, batch_size, t_dim_fast)

        if self.version == 'global_atten_fusion':
            task_predictions = self.global_attention_fusion(output_slow, output_fast, list_kf_indicies,
                                                            list_non_kf_indicies, mulit_task=self.multi_task)
        # if self.version == 'local_atten_fusion':
        #     task_predictions = self.local_attention_fusion(output_slow, output_fast, list_kf_indicies,
        #                                                    list_non_kf_indicies, mulit_task=self.multi_task)
        # last frame is a keyframe frame (slow encoded)
        if self.semi_sup['Mode']:
            all_encoded_frames = torch.cat([output_slow, output_fast], dim=1)
        index_of_labelled_frame = 0
        annotated_frame = {'last_frame': [], 'type': []}
        if max(list_kf_indicies) > max(list_non_kf_indicies):
            index_of_labelled_frame = len(list_kf_indicies) - 1
            if len(list_kf_indicies) > 1:
                annotated_frame['last_frame'].append(output_slow[:, -1].unsqueeze(1))
                annotated_frame['type'].append('slow_frame')
                output_slow = output_slow[:, :-1]
            else:
                annotated_frame['last_frame'].append(None)
                annotated_frame['type'].append('slow_frame')
        # last frame is a keyframe frame (fast encoded)
        else:
            index_of_labelled_frame = len(list_kf_indicies) + len(list_non_kf_indicies) - 1
            if len(list_non_kf_indicies) > 1:
                annotated_frame['last_frame'].append(output_fast[:, -1].unsqueeze(1))
                annotated_frame['type'].append('fast_frame')
                output_fast = output_fast[:, :-1]
            else:
                annotated_frame['last_frame'].append(None)
                annotated_frame['type'].append('fast_frame')

        # Different ways of propagating temporal features
        if self.version == 'average_fusion':
            task_predictions = self.average(output_slow, output_fast, annotated_frame, mulit_task=self.multi_task)
        elif self.version == 'advers':
            # still in development
            task_predictions = self.adverserial_concatentation(input, mulit_task=self.multi_task)
        elif self.version == 'conv3d_fusion':
            task_predictions = self.temporal_net(output_slow, output_fast, annotated_frame, mulit_task=self.multi_task)
        elif self.version == 'convnet_fusion':
            task_predictions = self.convnet_fusion(output_slow, output_fast, annotated_frame, with_se_block=False, mulit_task=self.multi_task)

        if self.semi_sup['Mode']:
            # take all the frames and run them through the main Seg decoder
            # all_encoded_frames = torch.cat([output_slow, output_fast], dim=1)
            # print(unlabelled_frame_fusion.shape)
            # unlabelled_pred will be a list of segmented outputs
            main_labelled_pred, main_unlabelled_pred, perturbed_unlabelled_pred = self.run_perturbed_decoders(all_encoded_frames, index_of_labelled_frame)

        # Output based on model choice
        if self.semi_sup['Mode'] and self.unsup['Mode']:
            return {'supervised': task_predictions, 'semi_supervised': [main_labelled_pred, main_unlabelled_pred, perturbed_unlabelled_pred], 'encoded_slow': x_slow_all, 'encoded_fast': x_fast_all}
        elif self.semi_sup['Mode'] and not self.unsup['Mode']:
            return {'supervised': task_predictions, 'semi_supervised': [main_labelled_pred, main_unlabelled_pred, perturbed_unlabelled_pred]}
        elif not self.semi_sup['Mode'] and self.unsup['Mode']:
            return {'supervised': task_predictions, 'encoded_slow': x_slow_all, 'encoded_fast': x_fast_all}
        else:
            return {'supervised': task_predictions}

    def compose_fusion_tensor(self, annotated_frame, output_slow, output_fast):
        x_fusion_input = torch.ones(output_slow.shape[0], output_slow.shape[1],
                                    output_slow.shape[2], output_slow.shape[3])
        if annotated_frame['type'][0] == 'slow_frame':
            if annotated_frame['last_frame'][0] is not None:
                x_fusion_input = torch.cat([output_slow, output_fast, annotated_frame['last_frame'][0]], dim=1)
            else:
                x_fusion_input = torch.cat([output_slow, output_fast[:, -1].unsqueeze(1)], dim=1)
        if annotated_frame['type'][0] == 'fast_frame':
            if annotated_frame['last_frame'][0] is not None:
                x_fusion_input = torch.cat([output_slow, output_fast, annotated_frame['last_frame'][0]], dim=1)
            else:
                x_fusion_input = torch.cat([output_slow, output_fast], dim=1)
        return x_fusion_input

    def average(self, output_slow, output_fast, annotated_frame, mulit_task=False):
        # output_slow = output_slow
        # fast_frame = output_fast[:, -1].unsqueeze(1)
        # x_fusion = torch.cat([output_slow, fast_frame], dim=1)
        x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        x_fusion = torch.mean(x_fusion_input, dim=1).squeeze(1)
        if not mulit_task:
            #  depth decoder or segmentation decoder
            task_predictions = self.task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
        else:
            # pass the average fusion to segmentation but not depth
            depth_pred = self.list_decoders[0](annotated_frame)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](x_fusion)
            seg_pred = seg_pred.squeeze(1)
            task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def temporal_net(self, output_slow, output_fast, annotated_frame, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        #  fast_frame = output_fast[:, -1].unsqueeze(1)
        # x_fusion = torch.cat([output_slow, fast_frame], dim=1).to('cuda:0')
        x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        # x_fusion = torch.cat([output_slow, fast_frame], dim=1)
        if not mulit_task:
            #  depth decoder or segmentation decoder
            # task_decoder = self.list_decoders[-1]
            task_predictions = self.task_decoder(x_fusion_input)
            task_predictions = task_predictions.squeeze(1)
        else:
            # pass the average fusion to segmentation but not depth
            depth_pred = self.list_decoders[0](annotated_frame)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](x_fusion)
            seg_pred = seg_pred.squeeze(1)
            task_predictions = [depth_pred, seg_pred]
        return task_predictions

    def convnet_fusion(self, output_slow, output_fast, annotated_frame, with_se_block=False, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        x_fusion_input = self.compose_fusion_tensor(annotated_frame, output_slow, output_fast)
        x_fusion = x_fusion_input.permute(0, 2, 1, 3, 4)
        x_fusion = self.convnet_fusion_layer(x_fusion)
        x_fusion = x_fusion.permute(0, 1, 2, 3, 4).squeeze()
        if with_se_block:
            x_fusion = x_fusion.unsqueeze(1)
            x_fusion = self.se_layer(x_fusion)
        if not mulit_task:
            #  depth decoder or segmentation decoder
            # task_decoder = self.list_decoders[-1]
            task_predictions = self.task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
        else:
            # pass the average fusion to segmentation but not depth
            depth_pred = self.list_decoders[0](annotated_frame)
            depth_pred = depth_pred.squeeze(1)
            seg_pred = self.list_decoders[1](x_fusion)
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

    # local attention if time permits, but does not work
    def local_attention_fusion(self, output_slow, output_fast, mulit_task=False):
        # take the slow output of the keyframes and the last frame of the fast
        # output_slow = output_slow.to('cuda:0')
        frame_batch_dim = output_slow.shape[0]
        frames_t_dim = output_slow.shape[1]
        frame_2d_batch = frame_batch_dim * frames_t_dim
        kf_encoded_frames = torch.reshape(output_slow, (frame_2d_batch, output_slow.shape[2], output_slow.shape[3],
                                                        output_slow.shape[4]))
        # print(kf_encoded_frames.shape)
        fast_frame = output_fast[:, -1]
        # print(fast_frame.shape)
        # fast_frame = fast_frame.tile((2,)).to('cuda:0')
        fast_frame = fast_frame.repeat(2, 1, 1, 1)
        # print(fast_frame.shape) - [4, 256, 8, 16]
        # print(kf_encoded_frames.shape) - [4, 256, 8, 16]
        # x_fusion = torch.cat([output_slow, fast_frame], dim=1).to('cuda:0')
        x_fusion = self.local_attention_block(kf_encoded_frames, fast_frame)
        print('After fusion attention: ' + str(x_fusion.shape))
        # conv_layer = nn.Conv3d(x_fusion.shape[1], 1, kernel_size=(1, 1, 1), stride=1).to('cuda:0')
        # x_fusion = conv_layer(x_fusion).squeeze()

        # if with_se_block:
        #     x_fusion = x_fusion.unsqueeze(1)
        #     se_block = ChannelSELayer3D(x_fusion.shape[1], 2).to('cuda:0')
        #     x_fusion = se_block(x_fusion)

        if not mulit_task:
            #  depth decoder or segmentation decoder
            # task_decoder = self.list_decoders[-1]
            task_predictions = self.task_decoder(x_fusion)
            task_predictions = task_predictions.squeeze(1)
            print('task_predictions.shape: ' + str(task_predictions.shape))
        else:
            task_predictions = []
            for task_decoder in self.list_decoders:
                task_pred = task_decoder(x_fusion)
                task_pred = task_pred.squeeze(1)
                task_predictions.append(task_pred)
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

    def reshape_output(self, input, batch_dim, t_dim):
        output = torch.reshape(input, (batch_dim, t_dim, input.shape[1],
                                       input.shape[2], input.shape[3]))
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


class MultiTaskModel1(nn.Module):
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
        print(input.shape)
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
