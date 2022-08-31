import os
import torch
import numpy as np
import imageio
import os
import torch
import numpy as np
import imageio
import argparse
import oyaml as yaml
import re
import torchvision.transforms as transforms
from torch.utils import data
import random
# from ptsemseg.utils import recursive_glob
# from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
# from augmentations import *
import cv2
from PIL import Image as im
import json
from PIL import Image
from collections import namedtuple


def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir, topdown=True)
        for filename in sorted(filenames)
        if filename.endswith(suffix)
    ]


def recursive_glob_set(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return set(
        filename[:-15]
        for looproot, _, filenames in os.walk(rootdir, topdown=True)
        for filename in sorted(filenames)
        if filename.endswith(suffix)
    )


class temporalLoader2(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(
            self,
            root,
            split="train",
            transform=None,
            test_mode=False,
            model_name=None,
            interval=5,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        # self.path_num = path_num
        self.window_size = interval
        # self.K = K
        self.root = root
        self.split = split
        self.transform = transform
        self.test_mode = test_mode
        self.model_name = model_name
        self.n_classes = 19
        self.files = {}
        self.seg_files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.videos_base = os.path.join(self.root, "leftImg8bit_sequence", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.depth_base = os.path.join(self.root, "disparity_sequence", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.seg_files[split] = recursive_glob_set(rootdir=self.images_base, suffix=".png")
        # print(self.files['train'][:50])
        # print(self.seg_files['train'])
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.depth_transform_train = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomHorizontalFlip(),
        ])
        self.depth_transform_val = transforms.Compose([
            transforms.Resize((128, 256)),
        ])
        self.image_transform_train = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.image_transform_val = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.lbl_transform_train = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomHorizontalFlip(),

        ])
        self.lbl_transform_val = transforms.Compose([
            transforms.Resize((128, 256)),
        ])

        self.ignore_index = 19
        self.class_map = dict(zip(self.valid_classes, range(19)))
        self.img_transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        images = list()
        segmentation_labels = list()
        depth_lbl = list()
        img_path_list = list()
        # normal_labels = list()
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        # lbl = imageio.imread(lbl_path)
        # lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        # seg_labels = torch.from_numpy(lbl).long()
        # seg_labels = seg_labels.reshape(1, seg_labels.shape[0], seg_labels.shape[1])
        vid_info = img_path.split('/')[-1].split('_')
        city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
        lbl_img = Image.open(lbl_path)
        image_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, int(cur_frame))))
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            _, label_target = self.transform(image, lbl_img)
            # label_target = self.lbl_transform_train(label_target)
            # label_target = torch.from_numpy(np.array(label_target, dtype='uint8'))
        # else:
        # label_target = self.lbl_transform_val(label_target)
        # label_target = torch.from_numpy(np.array(label_target, dtype='uint8'))
        label_target = self.encode_target(label_target)
        label_target = torch.from_numpy(label_target).long()
        annotated_frame_index = int(cur_frame)
        start_index = annotated_frame_index - (self.window_size - 1)
        end_index = annotated_frame_index + 1
        mask_interested_frames = [0, self.window_size-1, self.window_size]
        # print(annotated_frame_index)
        for index_frame in range(start_index, end_index):
            image_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, index_frame)))
            depth_path = os.path.join(self.depth_base, city, ("%s_%s_%06d_disparity.png" % (city, seq, index_frame)))
            # print(image_path)
            # image = imageio.imread(image_path)
            curr_img_path = "%s_%s_%06d_leftImg8bit" % (city, seq, index_frame)
            img_path_list.append(curr_img_path)
            image = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                image, _ = self.transform(image, lbl_img)
            # if self.split == 'train':
            #     image = self.image_transform_train(image)
            # else:
            #     image = self.image_transform_val(image)
            # image = np.array(image, dtype=np.uint8)
            # image = torch.from_numpy(image).permute(2, 0, 1).float()
            images.append(image)
            # images = torch.stack(image, dim=0)

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            disparity = torch.from_numpy(self.map_disparity(depth)).unsqueeze(0).float()
            depth_normalized = (1 - (-1)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
            if self.split == 'train':
                depth_labels = self.depth_transform_train(depth_normalized)
            else:
                depth_labels = self.depth_transform_val(depth_normalized)
            depth_lbl.append(depth_labels)

        # segmentation_labels = torch.stack(segmentation_labels, dim=0)
        depth_labels = torch.stack(depth_lbl, dim=0)
        images = torch.stack(images, dim=0)
        print(images.shape)
        print(depth_labels.shape)
        print(label_target.shape)
        print(img_path.shape)
        
        images = images[mask_interested_frames, ]
        depth_labels = depth_labels[mask_interested_frames, ]
        label_target = label_target[mask_interested_frames, ]
        img_path = img_path[mask_interested_frames, ]
        print(images.shape)
        print(depth_labels.shape)
        print(label_target.shape)
        print(img_path.shape)
        if self.split == 'val' or self.split == 'test':
            return images, label_target, depth_labels, img_path
        # print(images.shape) - torch.Size([5, 3, 128, 256])
        # print(label_target.shape) - torch.Size([128, 256])
        # print(depth_labels.shape) - torch.Size([5, 1, 128, 256])
        return images, label_target, depth_labels

    def map_disparity(self, disparity):
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        disparity[disparity == 0] = -1
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)
        return disparity

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)


class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]
    ]

    label_colours = dict(zip(range(20), colors))

    def __init__(
            self,
            root,
            split="train",
            augmentations=None,
            test_mode=False,
            model_name=None,
            frames_per_segment=30,
            path_num=2,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num = path_num
        self.interval = interval
        self.frames_per_segment = frames_per_segment
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.test_mode = test_mode
        self.model_name = model_name
        self.n_classes = 19
        self.files = {}
        self.seg_files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.videos_base = os.path.join(self.root, "leftImg8bit_sequence", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.depth_base = os.path.join(self.root, "disparity_sequence", self.split)

        self.files[split] = recursive_glob(rootdir=self.videos_base, suffix=".png")
        self.seg_files[split] = recursive_glob_set(rootdir=self.images_base, suffix=".png")
        # print(self.files[split][:500])
        self.frame_start_indices = self.get_start_indices()
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
            "unlabelled"
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.videos_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def get_start_indices(self):
        """ Gets the indices for the frames based on the number of frames in the dataset"""
        # choose start indices that are perfectly evenly spread across the video frames.
        num_segments = int(self.__len__() // self.frames_per_segment)

        if self.test_mode:
            distance_between_indices = (self.__len__() - self.frames_per_segment + 1) / float(num_segments)

            start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                      for x in range(num_segments)])
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (self.__len__() - self.frames_per_segment + 1) // num_segments

            start_indices = np.multiply(list(range(num_segments)), max_valid_start_index) + \
                            np.random.randint(max_valid_start_index, size=num_segments)

        return start_indices

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        # print( len(self.files[self.split]))
        next_start_idx = self.frame_start_indices[index]
        images = list()
        segmentation_labels = list()
        depth_labels = list()
        normal_labels = list()
        start_index = next_start_idx
        # consecutive frames
        frame_index = int(start_index)
        end_frame = self.__len__() - 1
        # create the sliding window for each annotated index

        # load self.frames_per_segment consecutive frames
        for _ in range(self.frames_per_segment):
            img_path = self.files[self.split][frame_index].rstrip()
            if img_path.split(os.sep)[-1][:-15] in self.seg_files[self.split]:
                lbl_path = os.path.join(
                    self.annotations_base,
                    img_path.split(os.sep)[-2],
                    os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
                )
                lbl = imageio.imread(lbl_path)
                lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
                seg_labels = torch.from_numpy(lbl).long()
                seg_labels = seg_labels.reshape(1, seg_labels.shape[0], seg_labels.shape[1])
                # segmentation_labels.append(seg_labels.reshape(1, seg_labels.shape[0], seg_labels.shape[1]))
                segmentation_labels.append(seg_labels)

            vid_info = img_path.split('/')[-1].split('_')
            city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]

            # image = self._load_image(record.path, frame_index)
            image_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq,
                                                                                               frame_index % 30)))
            depth_path = os.path.join(self.depth_base, city, ("%s_%s_%06d_disparity.png" % (city, seq,
                                                                                            frame_index % 30)))
            image = imageio.imread(image_path)
            image = np.array(image, dtype=np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            images.append(image)

            depth = imageio.imread(depth_path)
            depth = np.array(depth, dtype=np.uint8)
            depth = torch.from_numpy(depth).float()
            depth_reshaped = depth.reshape(1, depth.shape[0], depth.shape[1])
            depth_labels.append(depth_reshaped)

            if frame_index < end_frame:
                frame_index += 1

        # print(type(images[0]))
        images = torch.stack(images, dim=0)
        depth_labels = torch.stack(depth_labels, dim=0)
        segmentation_labels = torch.stack(segmentation_labels, dim=0)

        if self.augmentations is not None:
            images = self.augmentations(images)
            depth_labels = self.augmentations(depth_labels)
            segmentation_labels = self.augmentations(segmentation_labels)

        return images, depth_labels, segmentation_labels

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)


# key2aug = {
#     "rcrop": RandomCrop,
#     "hflip": RandomHorizontallyFlip,
#     "vflip": RandomVerticallyFlip,
#     "scale": Scale,
#     "rscale": RandomScale,
#     "rotate": RandomRotate,
#     "translate": RandomTranslate,
#     "ccrop": CenterCrop,
#     "colorjtr": ColorJitter,
#     "colornorm": ColorNorm
# }
#
# def get_composed_augmentations(aug_dict):
#     if aug_dict is None:
#         logger.info("Using No Augmentations")
#         return None
#
#     augmentations = []
#     for aug_key, aug_param in aug_dict.items():
#         augmentations.append(key2aug[aug_key](aug_param))
#         # logger.info("Using {} aug with params {}".format(aug_key, aug_param))
#     return Compose(augmentations)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    init_seed(11733, en_cudnn=False)
    print('starting training')
    # Setup Augmentations
    # train_augmentations = cfg["training"].get("train_augmentations", None)
    # t_data_aug = get_composed_augmentations(train_augmentations)
    # val_augmentations = cfg["validating"].get("val_augmentations", None)
    # v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader

    path_n = cfg["model"]["path_num"]

    # data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]
    train_augmentations = torch.nn.Sequential(
        transforms.Resize(size=(2048, 1024)),
        transforms.RandomCrop(size=(512, 1024)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.Normalize(mean=(123.675, 116.28, 103.53),
        #                      std=(58.395, 57.12, 57.375)),
        transforms.Pad(padding=(512, 1024)))
    t_loader = temporalLoader(data_path,
                              split=cfg["data"]["train_split"],
                              augmentations=train_augmentations,
                              path_num=path_n)
    # v_loader = cityscapesLoader(data_path, split=cfg["data"]["val_split"], augmentations=v_data_aug, path_num=path_n)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"],
                                  num_workers=cfg["training"]["n_workers"],
                                  shuffle=False,
                                  drop_last=True)

    # augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    # scripted_transforms = torch.jit.script(transforms)
    # local_path = "/home/zcqsspf/Scratch/data/cityscapes"
    # dst = cityscapesLoader(local_path, augmentations=augmentations)
    # bs = 4
    # trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, depth_labels, segmentation_labels = data_samples
        print(imgs.shape)
        print(depth_labels.shape)
        print(segmentation_labels.shape)