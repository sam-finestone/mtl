from collections import OrderedDict
from math import ceil, floor

import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class SlowFastLayers(nn.Module):
    def __init__(self, input_size, device, slow_pathway_size, fast_pathway_size):
        super(SlowFastLayers, self).__init__()
        self.device = device
        self.slow_pathway_size = slow_pathway_size
        self.fast_pathway_size = fast_pathway_size

        kernel_size_slow1, kernel_size_slow2, kernel_size_slow3 = self._calc_kernel_sizes(self.slow_pathway_size)
        kernel_size_fast1, kernel_size_fast2, kernel_size_fast3 = self._calc_kernel_sizes(self.fast_pathway_size)

        kernel_size_f2s1, slow_out1, fast_out1 = self._calc_fuse_kernel_size(slow_in=self.slow_pathway_size,
                                                                             slow_kernel=kernel_size_slow1,
                                                                             fast_in=self.fast_pathway_size,
                                                                             fast_kernel=kernel_size_fast1)
        kernel_size_f2s2, _, _ = self._calc_fuse_kernel_size(slow_in=slow_out1, slow_kernel=kernel_size_slow2,
                                                             fast_in=fast_out1, fast_kernel=kernel_size_fast2)

        self.fast_conv1, self.bn_f1 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_fast1,
                                                             in_channels=input_size, out_channels=32)

        self.slow_conv1, self.bn_s1 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_slow1,
                                                             in_channels=input_size, out_channels=192)

        self.fast_conv2, self.bn_f2 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_fast2,
                                                             in_channels=32, out_channels=32)

        self.slow_conv2, self.bn_s2 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_slow2,
                                                             in_channels=256, out_channels=192)

        self.fast_conv3, self.bn_f3 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_fast3,
                                                             in_channels=32, out_channels=32)

        self.slow_conv3, self.bn_s3 = self._init_conv_and_bn(temporal_kernelsize=kernel_size_slow3,
                                                             in_channels=256, out_channels=224)

        self.conv_f2s1, self.bn_f2s1 = self._init_fuse_and_bn(kernel_size_f2s1)

        self.conv_f2s2, self.bn_f2s2 = self._init_fuse_and_bn(kernel_size_f2s2)

        self.relu = nn.ReLU(inplace=True)

    def _init_conv_and_bn(self, temporal_kernelsize, in_channels, out_channels):
        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernelsize, 3, 3),
            padding=(0, 1, 1))

        bn = nn.BatchNorm3d(out_channels)

        return conv, bn

    def _init_fuse_and_bn(self, temporal_kernelsize):
        conv_f2s = nn.Conv3d(
            32,
            64,
            kernel_size=[temporal_kernelsize, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )

        bn_f2s = nn.BatchNorm3d(64)

        return conv_f2s, bn_f2s

    def _calc_kernel_sizes(self, pathway_size):
        div = pathway_size // 3
        if pathway_size % 3 == 0:
            return (div, div + 1, div + 1)
        elif pathway_size % 3 == 1:
            return (div + 1, div + 1, div + 1)
        elif pathway_size % 3 == 2:
            return (div + 1, div + 1, div + 2)

    def _calc_fuse_kernel_size(self, slow_in, slow_kernel, fast_in, fast_kernel):
        out_slow = (slow_in - slow_kernel) + 1
        out_fast = (fast_in - fast_kernel) + 1
        fuse_kernel_size = out_fast - out_slow + 1
        return fuse_kernel_size, out_slow, out_fast

    def fuse(self, slow, fast, conv, bn):
        fuse = conv(fast)
        fuse = bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([slow, fuse], 1)
        return x_s_fuse, fast

    def forward(self, slow, fast):
        # First Conv Layer
        slow = self.slow_conv1(slow)
        slow = self.bn_s1(slow)
        slow = self.relu(slow)

        fast = self.fast_conv1(fast)
        fast = self.bn_f1(fast)
        fast = self.relu(fast)

        # Fuse1
        slow, fast = self.fuse(slow, fast, self.conv_f2s1, self.bn_f2s1)

        # Second Conv Layer
        slow = self.slow_conv2(slow)
        slow = self.bn_s2(slow)
        slow = self.relu(slow)

        fast = self.fast_conv2(fast)
        fast = self.bn_f2(fast)
        fast = self.relu(fast)

        # Fuse2
        slow, fast = self.fuse(slow, fast, self.conv_f2s2, self.bn_f2s2)

        # Second Conv Layer
        slow = self.slow_conv3(slow)
        slow = self.bn_s3(slow)

        fast = self.fast_conv3(fast)
        fast = self.bn_f3(fast)
        return slow, fast

    def temporally_enhance_features(self, slow_features, fast_features):
        # List of dicts to dict of lists
        slow_features = {k: [dic[k] for dic in slow_features] for k in slow_features[0]}
        fast_features = {k: [dic[k] for dic in fast_features] for k in fast_features[0]}
        merged_features = OrderedDict()
        for key in slow_features.keys():
            key_scale_slow_features = torch.stack(slow_features[key]).to(self.device).transpose(1, 2)
            key_scale_fast_features = torch.stack(fast_features[key]).to(self.device).transpose(1, 2)
            key_scale_slow_features, key_scale_fast_features = self.forward(key_scale_slow_features,
                                                                            key_scale_fast_features)

            merged_features[key] = torch.cat([key_scale_slow_features, key_scale_fast_features], dim=1).squeeze(dim=2)
            del key_scale_slow_features, key_scale_fast_features

        return merged_features


class SegmentationModel(nn.Module):
    def __init__(self, device, slow_pathway_size, fast_pathway_size):
        super(SegmentationModel, self).__init__()
        self.device = device
        self.maskrcnn_model = get_model_instance_segmentation(num_classes=2)
        self.maskrcnn_model.load_state_dict(torch.load('maskrcnn/maskrcnn_model.pth'))

        # Freeze most of the weights
        for param in self.maskrcnn_model.backbone.parameters():
            param.requires_grad = False
        for param in self.maskrcnn_model.rpn.parameters():
            param.requires_grad = False

        self.slow_pathway_size = slow_pathway_size
        self.fast_pathway_size = fast_pathway_size

        self.slow_fast = SlowFastLayers(256, device=device, slow_pathway_size=slow_pathway_size,
                                        fast_pathway_size=fast_pathway_size)

        self.maskrcnn_model.roi_heads.detections_per_img = 10
        self.features_cache = {}
        self.use_caching = True  # as we do not train backbone we can reuse the features always

    def compute_maskrcnn_features(self, images_tensors, indices):  # TODO code review this function
        for key in list(self.features_cache.keys()):  # Delete features from cache that will never be used
            if key < indices[0]:
                self.features_cache.pop(key)

        all_features = OrderedDict()

        for idx in indices:
            if self.use_caching and idx in self.features_cache:
                features = self._detach_features(self.features_cache[idx])
            else:
                if idx >= 0 and idx < len(images_tensors):
                    batch_imgs = images_tensors[idx:idx + 1].to(self.device)
                    features = self.maskrcnn_model.backbone(batch_imgs)
                    if self.use_caching:
                        self.features_cache[idx] = features
                else:
                    continue
            for key, value in features.items():
                if key not in all_features:
                    all_features[key] = value
                else:
                    all_features[key] = torch.cat([all_features[key], value])

        left_pad_needed = len([elem for elem in indices if elem < 0])
        right_pad_needed = len([elem for elem in indices if elem >= len(images_tensors)])
        if left_pad_needed > 0:
            for key, image_feature in all_features.items():
                all_features[key] = torch.cat(
                    [torch.zeros_like(image_feature[:1, :, :, :].repeat(left_pad_needed, 1, 1, 1)), image_feature])

        if right_pad_needed > 0:
            for key, image_feature in all_features.items():
                all_features[key] = torch.cat(
                    [image_feature, torch.zeros_like(image_feature[:1, :, :, :].repeat(right_pad_needed, 1, 1, 1))])

        return all_features

    def batch_slice_features(self, features: OrderedDict, begin, end):
        batch_features = OrderedDict()
        for key, value in features.items():
            batch_features[key] = value[begin:end].to(self.device)

        return batch_features

    def compute_rpn_proposals(self, image_tensors, image_sizes, features, target):
        batch_imgs = ImageList(image_tensors.to(self.device), image_sizes)
        batch_proposals, proposal_loss = self.maskrcnn_model.rpn(batch_imgs, features, target)

        return batch_proposals, proposal_loss

    def _slice_features(self, features: OrderedDict, image_feature_idx, pathway_size):
        batch_features = OrderedDict()
        for key, value in features.items():
            batch_features[key] = value[image_feature_idx - floor(pathway_size / 2):image_feature_idx + ceil(
                pathway_size / 2)]

        return batch_features

    def _targets_to_device(self, targets, device):
        device_targets = []
        for i in range(len(targets)):
            device_target = OrderedDict()
            for key, value in targets[i].items():
                device_target[key] = value.to(device)

            device_targets.append(device_target)

        return device_targets

    def _index_features(self, features, i_begin, i_end):
        indexed_features = OrderedDict()
        for key, value in features.items():
            indexed_features[key] = features[key][i_begin:i_end]

        return indexed_features

    def _detach_features(self, features):
        indexed_features = OrderedDict()
        for key, value in features.items():
            indexed_features[key] = features[key].detach()

        return indexed_features

    def forward(self, images, targets=None, optimizer=None):
        # TODO test
        self.features_cache = {}
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        transformed_images, _ = self.maskrcnn_model.transform(images)

        '''Deal with imgs that have no objects in them'''
        valid_features_mask = []
        valid_targets = []
        valid_imgs = []
        for idx in range(len(transformed_images.tensors)):
            if 'boxes' not in targets[idx] or len(targets[idx]['boxes']) == 0:  # If no box predictions just skip
                valid_features_mask.append(0)
                continue
            else:
                valid_features_mask.append(1)
                valid_targets.append(targets[idx])
                valid_imgs.append(images[idx])

        _, targets = self.maskrcnn_model.transform(valid_imgs, valid_targets)

        del valid_imgs, valid_targets

        images = transformed_images
        full_targets = []
        pointer = 0
        for valid_id in valid_features_mask:
            if valid_id:
                full_targets.append(targets[pointer])
                pointer += 1
            else:
                full_targets.append({})

        targets = full_targets

        total_loss = 0.
        all_detections = []
        count = 0

        for feature_idx in range(len(valid_features_mask)):
            if valid_features_mask[feature_idx] != 1:
                continue

            padded_idx = self.fast_pathway_size // 2
            indices = range(feature_idx - floor(self.fast_pathway_size / 2), feature_idx + ceil(self.fast_pathway_size / 2))
            with torch.no_grad():
                image_features = self.compute_maskrcnn_features(transformed_images.tensors, indices)
            sliced_features = self._index_features(image_features, padded_idx, padded_idx + 1)
            target = targets[feature_idx:feature_idx + 1]
            target = self._targets_to_device(target, self.device)
            with torch.no_grad():
                rpn_proposals, proposal_loses = self.compute_rpn_proposals(transformed_images.tensors[feature_idx:feature_idx + 1],
                                                                           transformed_images.image_sizes[feature_idx:feature_idx + 1],
                                                                           sliced_features,
                                                                           target)

            target[0]['proposals'] = rpn_proposals[0]
            slow_valid_features = [
                self._slice_features(image_features, image_feature_idx=padded_idx, pathway_size=self.slow_pathway_size)]  # TODO test this well
            fast_valid_features = [image_features]

            slow_fast_features = self.slow_fast.temporally_enhance_features(slow_valid_features, fast_valid_features)
            batch_original_image_sizes = original_image_sizes[feature_idx:feature_idx + 1]
            batch_image_sizes = images.image_sizes[0:1] * len(
                batch_original_image_sizes)  # Because all images in one sequence have the same size
            proposals = [elem['proposals'] for elem in target]  # predicted boxes

            detections, detector_losses = self.maskrcnn_model.roi_heads(slow_fast_features, proposals, batch_image_sizes, target)
            detections = self.maskrcnn_model.transform.postprocess(detections, batch_image_sizes, batch_original_image_sizes)
            detections = self._targets_to_device(detections, device=torch.device('cpu'))
            all_detections.extend(detections)

            del slow_valid_features, fast_valid_features, target, slow_fast_features

            if self.training:
                losses = {}
                for key, value in detector_losses.items():
                    if key not in losses:
                        losses[key] = value
                    else:
                        losses[key] += value

                for key, value in proposal_loses.items():
                    if key not in losses:
                        losses[key] = value
                    else:
                        losses[key] += value

                losses = sum(loss for loss in losses.values())
                total_loss += losses.item()
                losses.backward()
                count += 1
                del losses
                if count % 2 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        # Append empty detection for non valid ids
        if not self.training:
            full_detections = []
            pointer = 0
            for valid_id in valid_features_mask:
                if valid_id:
                    full_detections.append(all_detections[pointer])
                    pointer += 1
                else:
                    full_detections.append({})

            all_detections = full_detections

        return (total_loss, all_detections)