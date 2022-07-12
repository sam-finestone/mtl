import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageOps
import time
import numpy as np
import copy
import random
import numbers
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import cv2
from matplotlib import pyplot as plt
# from train import *
import pandas as pd
import pickle
import shutil

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# read phase annotations from files into pandas DataFrame
def encode_labels(phase_path):
    videos_encoded_labels = []
    for i in range(len(phase_path)):
        video_phases = pd.read_csv(phase_path[i], sep='\t')
        le = LabelEncoder()
        video_phases['Phase'] = le.fit_transform(video_phases['Phase'])
        videos_encoded_labels.append(video_phases)
    return videos_encoded_labels

def split_datatset(path_dataset):
    videos_path = os.path.join(path_dataset, 'videos')
    tool_path = os.path.join(path_dataset, 'tool_annotations')
    phase_annotation = os.path.join(path_dataset, 'phase_annotations')
    data = dict()
    # loop through files in directory
    for filename in os.listdir(videos_path):
        # f = os.path.join(path_dataset, filename)
        # get all files .txt and .mp4
        current_file = filename[0:7]
        current_file_path = os.path.join(videos_path, filename)
        if filename.endswith('.mp4') and current_file not in data.keys():
            data[current_file] = {'video': {}, 'tools': {}, 'phase': {}}
            data[current_file]['video'] = current_file_path
    for filename in os.listdir(tool_path):
        current_file = filename[0:7]
        current_file_path = os.path.join(tool_path, filename)
        if current_file in data.keys():
            data[current_file]['tools'] = current_file_path

    for filename in os.listdir(phase_annotation):
        current_file = filename[0:7]
        current_file_path = os.path.join(phase_annotation, filename)
        if current_file in data.keys():
            data[current_file]['phase'] = current_file_path
    return data

def tool_to_int(df, row):
    #  Grasper: 0  Bipolar: 1  Hook: 2 Scissors: 3  Clipper: 4 Irrigator: 5 SpecimenBag: 6 None: 7
    for col in range(len(df.columns)):
       if df.iloc[row, col] == 1:
            return col
    return 7

def get_tool_name(tools_path_videos):
    dfs_correctly_labelled = []
    num_each = []
    for i in range(len(tools_path_videos)):
        df = pd.read_csv(tools_path_videos[i], sep='\t')
        tool_labels = []
        for row in range(len(df.index)):
            tool_labels.append(tool_to_int(df, row))
        df['tool'] = tool_labels
        dfs_correctly_labelled.append(df[['Frame', 'tool']])
        num_each.append((len(df.index) // 25) + 1)
    return dfs_correctly_labelled, num_each

def create_annotation_file(filename, frame_path, tools_path, phase_path):
    train_annotation = open(filename, 'w')
    for idx, video in enumerate(frame_path):
        video_name = video[32:]
        video_path = os.path.join('./dataset/cholec80/frame_resize', video_name)
        # print(tools_path[idx])
        count = 1
        # for loop through each frame for label (tool) + phase
        num_each = (len(tools_path[idx].index) // 25) + 1
        for frame_number in range(0, len(tools_path[idx].index), 25):
            print(frame_number)
            name = "frame%d.jpg" % frame_number
            VIDEO_PATH = os.path.join(video_path, video_name + '-' + name)
            # for every 25 frame then add the number
            if frame_number % 25 == 0 and frame_number < len(tools_path[idx].index):
                idx_for_tools = frame_number // 25
                tool_used = tools_path[idx].iloc[idx_for_tools, 1]
            train_annotation.write(
                VIDEO_PATH + ' ' + str(phase_path[idx].iloc[frame_number, 1]) + ' ' + str(tool_used) + ' ' + str(num_each) + ' '+ '\n')
    train_annotation.close()

# def create_annotation_file2(filename, frame_resize, tools_path, phase_path):
#     train_annotation = open(filename, 'w')
#     for filename in os.listdir(frame_resize):
#         video_name = filename[:-4]
#         img_frame_path = os.path.join(frame_resize, filename)
#         file_tool = [file for file in os.listdir(tools_path) if file.startswith(video_name)]
#         # for loop through each frame for label (tool) + phase
#         for frame_number in range(0, len(tools_path[idx].index), 25):
#             print(frame_number)
#             VIDEO_PATH = os.path.join(video_path, 'frame_' + str(frame_number))
#             # for every 25 frame then add the number
#             if frame_number % 25 == 0 and frame_number < len(tools_path[idx].index):
#                 idx_for_tools = frame_number // 25
#                 tool_used = tools_path[idx].iloc[idx_for_tools, 1]
#             train_annotation.write(
#                 VIDEO_PATH + ' ' + str(phase_path[idx].iloc[frame_number, 1]) + ' ' + str(tool_used) + '\n')
#     train_annotation.close()

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
                    resized_frame = cv2.resize(frame, (250, 250))
                    cv2.imwrite(frame_img_path, resized_frame)
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

def all_videos_to_frames(video_filepath, videos=3):
    root = './dataset/cholec80/videos'
    os.makedirs('./dataset/cholec80/frame_resize')
    count_video = 0

    for filename in os.listdir(video_filepath):
        if filename.endswith('.mp4') and videos >= count_video:
            file_path = os.path.join(root, filename)
            video_name = filename[:-4]
            dst_path = os.path.join('./dataset/cholec80/frame_resize', video_name)
            # # capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            count = 0
            retaining = 1
            os.makedirs(dst_path)
            capture = cv2.VideoCapture(file_path)
            # os.chdir(dst)
            while retaining:
                retaining, frame = capture.read()
                name = "frame%d.jpg" % count
                frame_img_path = os.path.join('./dataset/cholec80/frame_resize', video_name, video_name + '-' + name)
                print(frame_img_path)
                if retaining:
                    resized_frame = cv2.resize(frame, (250, 250))
                    cv2.imwrite(frame_img_path, resized_frame)
                    count += 1
            capture.release()
            cv2.destroyAllWindows()
            count_video += 1


def main():
    path_dataset = './dataset/cholec80/'
    # data = split_datatset(path_dataset)
    # randomly choose a 80-10-10 split of the surgical footage
    # num_videos = int(len(data))
    # num_train = int(0.8 * num_videos)
    num_train = 1
    # train = np.random.choice(num_videos, num_train, replace=False)
    train = [0, 1]
    # test = np.asarray([i for i in range(num_videos) if i not in train])
    # val = test[0:len(test) // 2]
    # test = test[len(test) // 2:-1]
    val = [2]
    test = [3]
    frames_train_paths = []
    tools_train_paths = []
    phase_train_paths = []
    frames_val_paths = []
    tools_val_paths = []
    phase_val_paths = []
    frames_test_paths = []
    tools_test_paths = []
    phase_test_paths = []
    for idx, video in enumerate(os.listdir('./dataset/cholec80/frame_resize')):
        if idx in train:
            frames_train_paths.append(os.path.join('./dataset/cholec80/frame_resize', video))
            phase_train_paths.append(os.path.join('./dataset/cholec80/phase_annotations', video + '-' + 'phase.txt'))
            tools_train_paths.append(os.path.join('./dataset/cholec80/tool_annotations', video + '-' + 'tool.txt'))
        if idx in val:
            frames_val_paths.append(os.path.join('./dataset/cholec80/frame_resize', video))
            phase_val_paths.append(os.path.join('./dataset/cholec80/phase_annotations', video + '-' + 'phase.txt'))
            tools_val_paths.append(os.path.join('./dataset/cholec80/tool_annotations', video + '-' + 'tool.txt'))
        if idx in test:
            frames_test_paths.append(os.path.join('./dataset/cholec80/frame_resize', video))
            phase_test_paths.append(os.path.join('./dataset/cholec80/phase_annotations', video + '-' + 'phase.txt'))
            tools_test_paths.append(os.path.join('./dataset/cholec80/tool_annotations', video + '-' + 'tool.txt'))

    print('train_paths  : {:6d}'.format(len(frames_train_paths)))
    print('valid_paths  : {:6d}'.format(len(frames_val_paths)))
    print('test_paths   : {:6d}'.format(len(frames_test_paths)))

    # create a train/val/test directory
    TRAIN_PATH = os.path.join('.', 'train')
    VAL_PATH = os.path.join('.', 'val')
    TEST_PATH = os.path.join('.', 'test')

    # creates individual files for train/val/test with their respectufl video frames
    # create_frame_file(TRAIN_PATH, train_paths)
    # create_frame_file(VAL_PATH, val_paths)
    # create_frame_file(TEST_PATH, test_paths)

    # get the videos file paths for each split
    # train_paths = [train_paths[i]['video'] for i in range(len(train_paths))]
    # val_paths = [val_paths[i]['video'] for i in range(len(val_paths))]
    # test_paths = [test_paths[i]['video'] for i in range(len(test_paths))]

    # read tool annotations from files into pandas DataFrame
    tools_train_paths, train_num_each = get_tool_name(tools_train_paths)
    tools_val_paths, val_num_each = get_tool_name(tools_val_paths)
    tools_test_paths, test_num_each = get_tool_name(tools_test_paths)
    print(train_num_each)
    phase_train_paths = encode_labels(phase_train_paths)
    phase_val_paths = encode_labels(phase_val_paths)
    phase_test_paths = encode_labels(phase_test_paths)

    all_info_all = []
    all_info_all.append(frames_train_paths)
    all_info_all.append(frames_val_paths)
    all_info_all.append(frames_test_paths)

    all_info_all.append(phase_train_paths)
    all_info_all.append(phase_val_paths)
    all_info_all.append(phase_test_paths)

    all_info_all.append(tools_train_paths)
    all_info_all.append(tools_val_paths)
    all_info_all.append(tools_test_paths)

    with open('cholec80.pkl', 'wb') as f:
        pickle.dump(all_info_all, f)

    # create an annotation files with video path and target labels
    train_filename = 'annotations_train.txt'
    # if not os.path.exists(train_filename):
    create_annotation_file(train_filename, frames_train_paths, tools_train_paths, phase_train_paths)
    val_filename = 'annotations_val.txt'
    # if not os.path.exists(val_filename):
    create_annotation_file(val_filename, frames_val_paths, tools_val_paths, phase_val_paths)
    test_filename = 'annotations_test.txt'
    # if not os.path.exists(test_filename):
    create_annotation_file(test_filename, frames_test_paths, tools_test_paths, phase_test_paths)



if __name__ == "__main__":
    #  My way for the dataloader
    main()
    # Try Yueming method for the dataloader
    # all_videos_to_frames('./dataset/cholec80/videos', videos=3)
    print('here')

