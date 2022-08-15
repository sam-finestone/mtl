import os
import torch
import numpy as np
import imageio
import argparse
import oyaml as yaml
import re

from torch.utils import data
import random
# from ptsemseg.utils import recursive_glob
# from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
# from augmentations import *

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


import os
import torch
import numpy as np
import imageio

from torch.utils import data
import random



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


class temporal_loader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
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
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
            self,
            root,
            split="train",
            augmentations=None,
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
        self.path_num = path_num
        self.interval = interval
        # self.K = K
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

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        self.seg_files[split] = recursive_glob_set(rootdir=self.images_base, suffix=".png")

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
            "unlabelled",
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

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])*self.interval

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        images = list()
        segmentation_labels = list()
        depth_labels = list()
        normal_labels = list()

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        lbl = imageio.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        seg_labels = torch.from_numpy(lbl).long()
        seg_labels = seg_labels.reshape(1, seg_labels.shape[0], seg_labels.shape[1])
        vid_info = img_path.split('/')[-1].split('_')
        city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
        annotated_frame_index = int(cur_frame)
        start_index = annotated_frame_index - self.interval + 1
        end_index = annotated_frame_index + 1
        for index_frame in range(start_index, end_index):
            curr_frame = annotated_frame - index_frame

            # Add the segmentation ground-truth
            segmentation_labels.append(seg_labels)

            image_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, curr_frame)))
            depth_path = os.path.join(self.depth_base, city, ("%s_%s_%06d_disparity.png" % (city, seq, curr_frame)))
            image = imageio.imread(image_path)
            image = np.array(image, dtype=np.uint8)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            images.append(image)

            depth = imageio.imread(depth_path)
            depth = np.array(depth, dtype=np.uint8)
            depth = torch.from_numpy(depth).float()
            depth_reshaped = depth.reshape(1, depth.shape[0], depth.shape[1])
            depth_labels.append(depth_reshaped)

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

class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
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
    ]

    label_colours = dict(zip(range(19), colors))



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
        self.seg_files ={}

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
            "unlabelled",
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
    t_loader = cityscapesLoader(data_path, split=cfg["data"]["train_split"], augmentations=train_augmentations, path_num=path_n)
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