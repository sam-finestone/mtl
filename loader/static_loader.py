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


class staticLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
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
    # print(id_to_train_id)
    id_to_train_id[np.where(id_to_train_id == 255)] = 19
    # print(id_to_train_id)
    valid_class = np.unique([c.train_id for c in classes if c.id >= 0])
    # print(valid_class)
    valid_class[np.where(valid_class == 255)] = 19
    print(valid_class)
    valid_class = list(valid_class)
    class_label = [c.name for c in classes if not (c.ignore_in_eval or c.id < 0)]
    class_label.append('void')
    print(class_label)

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
        self.root = root
        self.split = split
        self.transform = transform
        self.test_mode = test_mode
        self.model_name = model_name
        self.n_classes = 19
        self.files = {}
        self.seg_files = {}
        # self.colours = self.colors
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.depth_base = os.path.join(self.root, "disparity", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")
        # self.seg_files[split] = recursive_glob_set(rootdir=self.images_base, suffix=".png")
        # print(self.files[split][:500])
        # self.frame_start_indices = self.get_start_indices()
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        # self.valid_classes = [
        #     7,
        #     8,
        #     11,
        #     12,
        #     13,
        #     17,
        #     19,
        #     20,
        #     21,
        #     22,
        #     23,
        #     24,
        #     25,
        #     26,
        #     27,
        #     28,
        #     31,
        #     32,
        #     33,
        # ]
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
            "void"
        ]
        self.valid_classes = np.array([i for i in range(20)])
        # with open('/home/sam/project/loader/cityscape_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        # self.lb_colors = {el['trainId']: el['color'] for el in labels_info}
        self.num_classes = 19
        # print(len(self.lb_map))
        # https://github.com/lorenmt/auto-lambda/blob/24591b7ff0d4498b18bd4b4c85c41864bdfc800a/create_dataset.py#L173
        self.ignore_class = 19
        self.depth_transform_train = transforms.Compose([
                                                    transforms.Resize((128, 256)),
                                                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                                    # transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()
                                                 ])
        self.depth_transform_val = transforms.Compose([
                                                        transforms.Resize((128, 256)),
                                                        transforms.ToTensor()
                                                    ])
        # self.trans = transforms.Compose([
        #                                 ColorJitter(
        #                                     brightness = 0.5,
        #                                     contrast = 0.5,
        #                                     saturation = 0.5),
        #                                 HorizontalFlip(),
        #
        #                                 ])
        self.class_map = dict(zip(self.valid_classes, range(19)))

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

    @classmethod
    def unmap_disparity(cls, disparity):
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        # want to revrese this disparity = (1 - (-1)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
        disparity = (1 - (0)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
        disparity[disparity == -1] = 0
        disparity[disparity > -1] = (disparity[disparity > -1] + 1) * (128 * 4)
        # disparity = (1 - (0)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
        return disparity

    @classmethod
    def map_to_rgb(cls, disparity):
        # disparity = (244 - (-0)) *(disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
        disparity = np.round((disparity + 1) * 255 / 2) * 127.5
        return disparity

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        # print( len(self.files[self.split]))
        # load self.frames_per_segment consecutive frames
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        vid_info = img_path.split('/')[-1].split('_')
        city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
        frame_id = int(cur_frame)
        label_target = Image.open(lbl_path)

        # labels = np.array(lbl, dtype=np.int64)
        # label = labels[np.newaxis, :]
        # # segmentation_label = self.convert_labels(label)
        # labels = self.encode_segmap(labels)

        image_path = os.path.join(self.images_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, frame_id)))
        # image = imageio.imread(image_path)
        image_pil = Image.open(image_path).convert('RGB')
        # image = np.array(image)
        if self.transform is not None:
            image, label_target = self.transform(image_pil, label_target)
            # label_target = self.augmentations(label_target)
        label_target = self.encode_target(label_target)
        # lbl = imageio.imread(lbl_path)
        # lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        label_target = torch.from_numpy(label_target).long()
        # label_target = label_target.reshape(1, label_target.shape[0], label_target.shape[1])

        # image = torch.from_numpy(image).permute(2, 0, 1).float()

        depth_path = os.path.join(self.depth_base, city, ("%s_%s_%06d_disparity.png" % (city, seq, frame_id)))
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_img = Image.fromarray(depth_img)
        # depth_img_tensor = torch.from_numpy(depth_img).float()
        # depth_img = Image.open(depth_path)
        # depth_np = np.asarray(depth, dtype=np.uint8) # without dtype for display

        # disparity = torch.from_numpy(self.map_disparity(depth)).unsqueeze(0).float()
        # depth_normalized = (1 - (-1)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1

        # print(depth_normalized.max()) - 1.0
        # print(depth_normalized.min()) - -1.0
        if self.split == 'train':
            depth_labels = self.depth_transform_train(depth_img)
            depth_normalized = self.map_disparity(depth_labels.float()).float()
        else:
            depth_labels = self.depth_transform_val(depth_img)
            depth_normalized = self.map_disparity(depth_labels.float()).float()

        if self.split == 'val' or self.split == 'test':
            # _, depth_labels = self.transform(image_pil, depth_img)
            # depth_labels = self.depth_transform_val(depth_img)
            # # depth_labels = transforms.ToTensor()(depth_labels)
            # # depth_labels[depth_labels == 0] = -1
            # # disparity = torch.from_numpy(self.map_disparity(depth_labels.float())).unsqueeze(0).float()
            # depth_normalized = self.map_disparity(depth_labels.float()).float()
            # # depth_normalized = (1 - (-1)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
            # # depth_normalized = depth_labels.float()
            img_path = "%s_%s_%06d_leftImg8bit" % (city, seq, frame_id)
            return image, label_target, depth_normalized, img_path

        # depth_labels = self.depth_transform_train(depth_img_tensor)
        # depth_labels = self.depth_transform_train(depth_img)
        # print(depth_labels.shape)
        # depth_labels = transforms.ToTensor()(depth_labels)
        # depth_labels = np.asarray(depth_labels, dtype="float32")
        # _, depth_labels = self.transform(image_pil, depth_img)
        # disparity = torch.from_numpy(self.map_disparity(depth_labels.float())).unsqueeze(0).float()
        # print(depth_labels.shape)
        # depth_normalized = self.map_disparity(depth_labels.float()).float()
        # print(depth_normalized.shape)

        # depth_normalized = (1 - (-1)) * (disparity - disparity.min()) / (disparity.max() - disparity.min()) - 1
        # depth_normalized = depth_labels.float()
        # print(depth_labels.shape)
        # print(label_target.shape)
        return image, label_target, depth_normalized

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

    def map_disparity(self, disparity):
        # https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        disparity[disparity == 0] = -1
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)
        return disparity

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    def preprocess(self, image, mask, flip=False, scale=None, crop=None):
        if flip:
            if random.random() < 0.5:
              image = image.transpose(Image.FLIP_LEFT_RIGHT)
              mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if scale:
            w, h = image.size
            rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
            random_scale = math.pow(2, rand_log_scale)
            new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
            image = image.resize(new_size, Image.ANTIALIAS)
            mask = mask.resize(new_size, Image.NEAREST)

        data_transforms = transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])
        image = data_transforms(image)
        mask = torch.LongTensor(np.array(mask).astype(np.int64))

        if crop:
            h, w = image.shape[1], image.shape[2]
            pad_tb = max(0, crop[0] - h)
            pad_lr = max(0, crop[1] - w)
            image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
            mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

            h, w = image.shape[1], image.shape[2]
            i = random.randint(0, h - crop[0])
            j = random.randint(0, w - crop[1])
            image = image[:, i:i + crop[0], j:j + crop[1]]
            mask = mask[i:i + crop[0], j:j + crop[1]]

        return image, mask
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
    # augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])
    DATASET_PATH = '/mnt/c44578a3-8e98-4fc3-a8b4-72266618fb8a/sam_dataset/data/cityscapes_static'
    example_depth = os.path.join(DATASET_PATH, 'disparity/train/aachen/aachen_000000_000019_disparity.png')
    ex_depth = imageio.imread(example_depth)
    plt.imshow(ex_depth)
    plt.savefig('/home/sam/project/img_depth.png')

    train_augmentations = torch.nn.Sequential(transforms.Resize(size=(128, 256))
                                              # transforms.RandomCrop(size=(256, 512)),
                                              # transforms.RandomHorizontalFlip(p=0.5),
                                              # transforms.Normalize(mean=(123.675, 116.28, 103.53),
                                              #                      std=(58.395, 57.12, 57.375)),
                                              # transforms.Pad(padding=(256, 512))
                                              )
    train_set = staticLoader(DATASET_PATH,
                              split='train',
                              augmentations=train_augmentations,
                              test_mode=False,
                              model_name=None, path_num=2)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=4,
                                                   shuffle=False,
                                                   num_workers=2,
                                                   drop_last=True)

    for batch_idx, (inputs, labels, depth) in enumerate(train_dataloader):
        plt.figure(figsize=(10, 10))
        # print(inputs[0].shape)
        # print(labels[0].shape)
        # print(depth[0].shape)
        # print(torch.max(inputs[0]))
        # print(torch.min(inputs[0]))
        #
        # print(torch.max(labels[0]))
        # print(torch.min(labels[0]))
        #
        # print(torch.max(depth[0]))
        # print(torch.min(depth[0]))
        print(inputs.shape)
        print(labels.shape)
        print(depth.shape)
        plt.imshow(labels[0].squeeze())
        # data.save('/home/sam/project/depth.png')
        plt.savefig('/home/sam/project/seg_gt.png')
        imgs = inputs.data.numpy()
        img_input = np.transpose(imgs[0], (1, 2, 0))
        cv2.imwrite('/home/sam/project/img.png', img_input)
        # lbl_input = np.transpose(lbl[0], (1, 2, 0))
        # print(labels.shape)
        # cv2.imwrite('/home/sam/project/seg.png', labels[0, :, :].squeeze().numpy())

        # print(np.transpose(inputs[0].data.numpy(), (1, 2, 0)).shape)
        # plt.imshow(np.transpose(inputs[0].data.numpy(), (1, 2, 0)))
        # plt.savefig('/home/sam/project/img_rbg.png')
        # plt.imshow(depth[0].numpy().squeeze())
        # plt.savefig('/home/sam/project/img_depth.png')
        # depth = depth.data.numpy()
        # img_input = np.transpose(depth[0], (1, 2, 0))
        # cv2.imwrite('/home/sam/project/img.png', img_input)

        plt.imshow(depth[0].squeeze())
        # data.save('/home/sam/project/depth.png')
        plt.savefig('/home/sam/project/depth.png')
        break


    # img = train_features[0].squeeze()
    # label = seg_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
    #
    # datapath = '~/Users/finess2/Desktop/ucf101/UCF101/UCF-101'
    # train_dataloader = DataLoader(VideoDataset(datapath, mode='train'), batch_size=10, shuffle=True, num_workers=0)
    #

    print('finished')