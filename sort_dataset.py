import pandas as pd
from preprocessing import *
import os
from PIL import Image, ImageOps
import time
import pickle
import numpy as np
import shutil
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pickle

def tool_to_int(df, row):
    #  Grasper: 0  Bipolar: 1  Hook: 2 Scissors: 3  Clipper: 4 Irrigator: 5 SpecimenBag: 6 None: 7
    for col in range(len(df.columns)):
       if df.iloc[row, col] == 1:
            return col
    return 7

def get_tool_name(tools_path_videos):
    dfs_correctly_labelled = []
    for i in range(len(tools_path_videos)):
        df = pd.read_csv(tools_path_videos[i], sep='\t')
        tool_labels = []
        for row in range(len(df.index)):
            tool_labels.append(tool_to_int(df, row))
        df['tool'] = tool_labels
        dfs_correctly_labelled.append(df[['Frame', 'tool']])
    return dfs_correctly_labelled

def create_annotation_file(filename, videos_path, tools_path, phase_path):
    train_annotation = open(filename, 'w')
    for idx, train_example in enumerate(videos_path):
        video_path = train_example[26:]
        video_path = os.path.join('.', 'train', video_path[:-4])
        print(tools_path[idx])
        # for loop through each frame for label (tool) + phase
        for frame_number in range(0, len(tools_path[idx].index), 25):
            print(frame_number)
            VIDEO_PATH = os.path.join(video_path, 'frame_' + str(frame_number))
            # for every 25 frame then add the number
            if frame_number % 25 == 0 and frame_number < len(tools_path[idx].index):
                idx_for_tools = frame_number // 25
                tool_used = tools_path[idx].iloc[idx_for_tools, 1]
            train_annotation.write(
                VIDEO_PATH + ' ' + str(phase_path[idx].iloc[frame_number, 1]) + ' ' + str(tool_used) + '\n')
    train_annotation.close()

def create_frame_file(dir_path, video_paths):
    split_name = dir_path[2:]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        for idx, video in enumerate(video_paths):
            video_name = video['video'][26:]
            src = video['video']
            dst_path = os.path.join('.', split_name, video_name)
            shutil.copy(src, dst_path)
            os.makedirs(dst_path[:-4])
            capture = cv2.VideoCapture(dst_path)
            # capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            count = 0
            retaining = 1
            # os.chdir(dst)
            while retaining:
                retaining, frame = capture.read()
                name = "frame%d.jpg" % count
                frame_img_path = os.path.join('.', split_name, video_name[:-4], name)
                if retaining and count % 25 == 0:
                    print(frame_img_path)
                    cv2.imwrite(frame_img_path, frame)
                count += 1
            capture.release()
            cv2.destroyAllWindows()
            # os.chdir('~/Desktop/project/mtl')

class CholecDataset(Dataset):
    def __init__(self, annotation_pathfile, transform=None):
        # loop through all train file
        self.img_frame_data = []
        file1 = open(annotation_pathfile, 'r')
        videos = file1.readlines()
        for frames in videos:
            data = frames.split()
            with open(data[0], 'rb') as f:
                with Image.open(f) as img:
                    self.img_frame_data.append(img.convert('RGB'))
            self.img_frame_phase.append(data[1])
            self.img_frame_tool.append(data[2])
        self.transform = transform

    def __getitem__(self, index):
        img_data = self.img_frame_data[index]
        labels_phase = self.file_labels_phase[index]
        labels_tool = self.file_labels_tool[index]
        # imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(img_data)

        return imgs, labels_phase, labels_tool

    def __len__(self):
        return len(self.img_frame_data)

def all_videos_to_frames(video_filepath):
    root = './dataset/cholec80/videos'
    os.makedirs('./dataset/cholec80/frame_resize')
    for filename in os.listdir(video_filepath):
        if filename.endswith('.mp4'):
            file_path = os.path.join(root, filename)
            video_name = filename[0:-4]
            # dst_path = os.path.join('.', 'dataset', 'cholec80', 'frames_resize')
            # # capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            count = 0
            retaining = 1
            capture = cv2.VideoCapture(file_path)
            # os.chdir(dst)
            while retaining:
                retaining, frame = capture.read()
                name = "frame%d.jpg" % count
                frame_img_path = os.path.join('./dataset/cholec80/frame_resize', video_name + '-' + name)
                print(frame_img_path)
                if retaining:
                    resized_frame = cv2.resize(frame, (250, 250))
                    cv2.imwrite(frame_img_path, resized_frame)
                    count += 1
            capture.release()
            cv2.destroyAllWindows()


class cholecDataset(Dataset):
    def __init__(self, video_paths, tool_lst, phase_df, transform=None, loader=pil_loader, clip_len = 8, frame_sample_rate = 1):
        self.video_paths = video_paths
        self.tool_labels = tool_lst
        self.phase_labels = phase_df
        self.transform = transform
        self.frame_sample_rate = frame_sample_rate
        self.clip_len = clip_len
        self.short_side = [128, 160]
        self.crop_size = 112
        # self.target_transform=target_transform
        self.loader = loader

    def __getitem__(self, index):
        curr_video = self.video_paths[index]
        labels_1 = self.phase_labels[index]
        # convert a df to tensor to be used in pytorch
        labels_1 = torch.from_numpy(labels_1.values).int()
        labels_2 = self.tool_labels[index]
        labels_2 = torch.from_numpy(labels_2.values).int()
        # imgs = self.loader(img_names)
        curr_imgs = self.loadvideo(curr_video)
        print(curr_imgs.shape)
        # if self.transform is not None:
        #     curr_imgs = self.transform(curr_imgs)
        print(curr_imgs.shape)
        # while curr_imgs.shape[0] < self.clip_len + 2:
        #     index = np.random.randint(self.__len__())
        #     curr_imgs = self.loadvideo(self.video_paths[index])
        # curr_imgs = self.crop(curr_imgs, self.clip_len, self.crop_size)
        curr_imgs = self.normalize(curr_imgs)
        curr_imgs = self.to_tensor(curr_imgs)
        print(curr_imgs.shape)
        # # print(labels_1)
        # one_img_test = curr_imgs[:, 0, :, :]
        # print(one_img_test.shape)
        # plt.imshow(one_img_test, interpolation='nearest')
        # plt.show()
        # print(d)
        return curr_imgs, (labels_1, labels_2)
        # if self.transform is not None:
        #     curr_imgs = self.transform(curr_imgs)
        # return curr_imgs, labels_1, labels_2

    def __len__(self):
        return len(self.video_paths)

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname, start_frame_number=50):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # if frame_height < frame_width:
        #     resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
        #     resize_width = int(float(resize_height) / frame_height * frame_width)
        # else:
        #     resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
        #     resize_height = int(float(resize_width) / frame_width * frame_height)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        # this basically takes 300 frames from the video at random to take
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, frame_height, frame_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0
        # read in each frame, one at a time into the numpy array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                # if (frame_height != resize_height) or (frame_width != resize_width):
                #     frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        # print(count)# 25,292 (or 73126) frames in video 1 of training
        capture.release()
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



