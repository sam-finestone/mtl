import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.utils.data.sampler as sampler
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, RandomShortSideScale, \
    ShortSideScale, Normalize
from torch import nn
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, CenterCrop
import os
from torch.utils.data import DataLoader
from pathlib import Path
import fnmatch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shutil
import fnmatch
import cv2

class Cityscape(torch.utils.data.Dataset):
  def __init__(self, directory_main):
    super(Cityscape, self).__init__()
    dir_images = os.path.join(directory_main, 'image')
    dir_dept = os.path.join(directory_main, 'depth')
    dir_labels = os.path.join(directory_main, 'label_7')
    # store the raw tensors
    self.images = []
    self.labels = []
    for filename in os.listdir(dir_images)[:150]:
        filename = os.path.join(dir_images, filename)
        self.images.append(np.load(filename))
    for filename in os.listdir(dir_labels)[:150]:
        filename = os.path.join(dir_labels, filename)
        self.labels.append(np.load(filename))
    # for filename in os.listdir(dir_dept)[:150]:
    #     filename = os.path.join(dir_dept, filename)
    #     self.depth.append(np.load(filename))


  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img = self.images[index]
    # depth  = self.depth[index]
    labels = self.labels[index]
    return img, labels

class CityScapes(Dataset):
  def __init__(self, root, train=True):
    self.train = train
    self.root = os.path.expanduser(root)

    # read the data file
    if train:
      self.data_path = root + '/train'
    else:
      self.data_path = root + '/val'
    self.frames_per_segment = 10
    self.sequence_length = 5
    # calculate data length
    # self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
    self.data_len = 20
    self.images = []
    self.semantic_labels = []
    self.depth_labels = []
    self.frame_sample_rate = 1
    self.clip_len = 8

  def __getitem__(self, index):
    # list_each_length = [self.data_len]
    # image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
    # semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
    # depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))

    # while buffer.shape[0] < self.clip_len + 2:
    #   index = np.random.randint(self.__len__())
    #   buffer = self.create_buffer(self)
    images, semantics, depth = self.create_buffer()
    images = self.to_tensor(images)
    # semantics = self.to_tensor(semantics)
    depth = self.to_tensor(depth)
    return images, semantics, depth

  def to_tensor(self, buffer):
    # convert from [D, C, H, W] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose((1, 0, 2, 3))

  def create_buffer(self):
  #   create a depth dimension to load the videos [C, D, H, W]
    remainder = np.random.randint(self.frame_sample_rate)
    # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
    start_idx = 0
    frame_count = self.data_len
    end_idx = frame_count - 1
    # [1, 3, 128, 256]
    frame_count_sample = frame_count // self.frame_sample_rate - 1
    if frame_count > 300:
      end_idx = np.random.randint(300, frame_count)
      start_idx = end_idx - 300
      frame_count_sample = 301 // self.frame_sample_rate - 1

    buffer_img = np.empty((frame_count_sample, 3, 128, 256), np.dtype('float32'))
    buffer_semantics = np.empty((frame_count_sample, 128, 256), np.dtype('float32'))
    buffer_depth = np.empty((frame_count_sample, 1, 128, 256), np.dtype('float32'))

    count = 0
    retaining = True
    sample_count = 0

    # read in each frame, one at a time into the numpy buffer array
    while (count <= end_idx):
      frame = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(count)), -1, 0)).float()
      semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(count))).float()
      depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(count)), -1, 0)).float()
      if count < start_idx:
        count += 1
        continue
      if count > end_idx:
        break
      if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
        # will resize frames if not already final size
        # print(frame.transpose(1, 2, 0))
        # print(buffer_img.shape)
        buffer_img[sample_count] = frame
        buffer_semantics[sample_count] = semantic
        buffer_depth[sample_count] = depth
        sample_count = sample_count + 1
      count += 1
    # print(sample_count)
    return buffer_img, buffer_semantics, buffer_depth

  def __len__(self):
    return self.data_len



class VideoDataset(Dataset):

  def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
    # folder = Path(directory) / mode  # get the directory of the specified split
    self.clip_len = clip_len
    self.data_path = directory
    self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))
    self.short_side = [128, 160]
    self.crop_size = 112
    self.frame_sample_rate = frame_sample_rate
    self.mode = mode

    self.fnames, labels,  = [], []
      #   for label in sorted(os.listdir(folder)):
      # for fname in os.listdir(os.path.join(folder, label)):
      #   self.fnames.append(os.path.join(folder, label, fname))
      #   labels.append(label)



    # # prepare a mapping between the label names (strings) and indices (ints)
    # self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
    # # convert the list of label names into an array of label indices
    # self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
    #
    # label_file = str(len(os.listdir(folder))) + 'class_labels.txt'
    # with open(label_file, 'w') as f:
    #   for id, label in enumerate(sorted(self.label2index)):
    #     f.writelines(str(id + 1) + ' ' + label + '\n')

  def __getitem__(self, index):
    # loading and preprocessing. TODO move them to transform classes
    buffer = self.load_images(self.fnames[index])
    image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
    semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
    depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))

    while buffer.shape[0] < self.clip_len + 2:
      index = np.random.randint(self.__len__())
      buffer = self.load_images(self.fnames[index])

    if self.mode == 'train' or self.mode == 'training':
      buffer = self.randomflip(buffer)
    buffer = self.crop(buffer, self.clip_len, self.crop_size)
    buffer = self.normalize(buffer)
    buffer = self.to_tensor(buffer)

    return buffer, semantic.float(),

  def to_tensor(self, buffer):
    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose((3, 0, 1, 2))

  def load_images(self, fname):
    remainder = np.random.randint(self.frame_sample_rate)
    # initialize a VideoCapture object to read video data into a numpy array
    # frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = self.data_len
    frame_width = 256
    frame_height = 128

    if frame_height < frame_width:
      resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
      resize_width = int(float(resize_height) / frame_height * frame_width)
    else:
      resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
      resize_height = int(float(resize_width) / frame_width * frame_height)

    # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
    start_idx = 0
    end_idx = frame_count - 1
    frame_count_sample = frame_count // self.frame_sample_rate - 1
    if frame_count > 300:
      end_idx = np.random.randint(300, frame_count)
      start_idx = end_idx - 300
      frame_count_sample = 301 // self.frame_sample_rate - 1
    buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

    count = 0
    retaining = True
    sample_count = 0

    # read in each frame, one at a time into the numpy buffer array
    index = 0
    while (count <= end_idx and index < frame_count):
      frame = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
      index += 1
      if count < start_idx:
        count += 1
        continue
      if count > end_idx:
        break
      if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # will resize frames if not already final size
        # if (frame_height != resize_height) or (frame_width != resize_width):
          # frame = cv2.resize(frame, (resize_width, resize_height))
        buffer[sample_count] = frame
        sample_count = sample_count + 1
      count += 1
    # capture.release()
    return buffer

  def crop(self, buffer, clip_len, crop_size):
    # randomly select time index for temporal jittering
    time_index = np.random.randint(buffer.shape[0] - clip_len)
    # Randomly select start indices in order to crop the video
    height_index = np.random.randint(buffer.shape[1] - crop_size)
    width_index = np.random.randint(buffer.shape[2] - crop_size)

    # crop and jitter the video using indexing. The spatial crop is performed on
    # the entire array, so each frame is cropped in the same location. The temporal
    # jitter takes place via the selection of consecutive frames
    buffer = buffer[time_index:time_index + clip_len,
             height_index:height_index + crop_size,
             width_index:width_index + crop_size, :]

    return buffer

  def normalize(self, buffer):
    # Normalize the buffer
    # buffer = (buffer - 128)/128.0
    for i, frame in enumerate(buffer):
      frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
      buffer[i] = frame
    return buffer

  def randomflip(self, buffer):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
    if np.random.random() < 0.5:
      for i, frame in enumerate(buffer):
        buffer[i] = cv2.flip(frame, flipCode=1)

    return buffer

  def __len__(self):
    return len(self.fnames)

class PackPathway(nn.Module):
  """
  Transform for converting video frames as a list of tensors.
  """

  def __init__(self, alpha=4):
    super().__init__()
    self.alpha = alpha

  def forward(self, frames):
    fast_pathway = frames
    # perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(frames, 1,
                                      torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long())
    frame_list = [slow_pathway, fast_pathway]
    return frame_list


def main():
  root = './dataset/cityscape/train'
  root2 = './dataset/data'
  # data prepare
  side_size = 256
  max_size = 320
  mean = [0.45, 0.45, 0.45]
  std = [0.225, 0.225, 0.225]
  crop_size = 256
  num_frames = 32
  sampling_rate = 2
  frames_per_second = 30
  clip_duration = (num_frames * sampling_rate) / frames_per_second
  num_classes = 19

  # train_transform = ApplyTransformToKey(key="video", transform=Compose(
  #   [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
  #    RandomShortSideScale(min_size=side_size, max_size=max_size), RandomCrop(crop_size), RandomHorizontalFlip(),
  #    PackPathway()]))
  #
  # test_transform = ApplyTransformToKey(key="video", transform=Compose(
  #   [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
  #    ShortSideScale(size=side_size), CenterCrop(crop_size), PackPathway()]))
  # train_data = labeled_video_dataset('{}/train'.format('./dataset/data/'), make_clip_sampler('random', clip_duration),
  #                                    transform=train_transform, decode_audio=False)
  # test_data = labeled_video_dataset('{}/val'.format('./dataset/data/'),
  #                                   make_clip_sampler('constant_clips_per_video', clip_duration, 1),
  #                                   transform=test_transform, decode_audio=False)
  # test_transform = ApplyTransformToKey(key="video", transform=Compose(
  #   [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
  #    ShortSideScale(size=side_size), CenterCrop(crop_size), PackPathway()]))

  # train_loader = DataLoader(train_data, batch_size=2, num_workers=2)
  # test_loader = DataLoader(test_data, batch_size=2, num_workers=2)

  # train_dataloader = Cityscape(root)
  # train = torch.utils.data.DataLoader(train_dataloader, batch_size=2, shuffle=True)
  train_dataloader = CityScapes(root2,  train=True)
  train = torch.utils.data.DataLoader(train_dataloader, batch_size=2, shuffle=True)

  # Display image and label.
  train_features, seg_labels, depth_labels = next(iter(train_dataloader))
  # train_features, seg_labels = next(iter(train_loader))
  # print(f"Feature batch shape: {train_features.size()}")
  # print(f"Labels batch shape: {train_labels.size()}")
  img = train_features[0].squeeze()
  label = seg_labels[0]
  plt.imshow(img, cmap="gray")
  plt.show()
  print(f"Label: {label}")

  datapath = '~/Users/finess2/Desktop/ucf101/UCF101/UCF-101'
  train_dataloader = DataLoader(VideoDataset(datapath, mode='train'), batch_size=10, shuffle=True, num_workers=0)
  for step, (buffer, label) in enumerate(train_dataloader):
    print("label: ", label)



if __name__ == "__main__":
    main()
    print('finished')