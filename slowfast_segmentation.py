import os
import time
import numpy as np
import sys
import torch
import argparse
import logging
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
import slowfast_mtl_model
from tensorboardX import SummaryWriter
# import itertools
from sort_dataset import *
from torchvision import models, transforms
from city_dataloader import CityScapes
from utils import *
from decoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

# commandline arguments
parser = argparse.ArgumentParser(description='cnn_lstm training')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-d', '--dataset_path', default='./dataset/data/', type=str, help='dataset path, default home directory')
parser.add_argument('-s', '--seq', default=2, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train_batch', default=1, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val_batch', default=1, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=1, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=1, type=int, help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-3, type=float, help='learning rate for optimizer, default 1e-3')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd, default 0.9')
parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay for sgd, default 0')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd, default 0')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
parser.add_argument('--sgdstep', default=5, type=int, help='number of steps to adjust lr for sgd, default 5')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
parser.add_argument('--seed', type=int, default=123, help='random seed number for reproducibility')
parser.add_argument('--sg_num', type=int, default=7, help='number of classes for segmentation')
parser.add_argument('--pretrain', type=bool, default=None, help='Pretrain model')
parser.add_argument('--model_save_path', type=str, default='.', help='Path to the saved models')
parser.add_argument('--window_size', type=int, default=5, help='ILA module computes on a window size')
parser.add_argument('--kf_intervals', type=int, default=5, help='Key-frame intervals')

args = parser.parse_args()

torch.manual_seed(args.seed)
gpu_usg = ",".join(list(map(str, args.gpu)))
# sequence_length = args.seq
DATASET_PATH = args.dataset_path
TRAIN_BATCH_SIZE = args.train_batch
VAL_BATCH_SIZE = args.val_batch
# optimizer_choice = args.opt
# multi_optim = args.multi
EPOCHS = args.epo
WORKERS = args.work
# use_flip = args.flip
# crop_type = args.crop
LR = args.lr
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weightdecay
# dampening = args.dampening
# use_nesterov = args.nesterov
NUMBER_CLASSES = args.sg_num
PRE_TRAIN = args.pretrain
sgd_adjust_lr = args.sgdadjust
SGD_STEP = args.sgdstep
SGD_GAMMA = args.sgdgamma
L = args.window_size

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

config = { 'temp': 2.0, 'weight': 'dwa'}

# Load the dataloaders
print('preprocessing dataloader')
train_set = CityScapes(root=DATASET_PATH, train=True)
val_set = CityScapes(root=DATASET_PATH, train=False)
train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=WORKERS)
test_dataloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=WORKERS)

encoder = slowfast.resnet50().to(device)
# wandb.watch(encoder)
# optimizer = optim.Adam(model.parameters(), lr=LR)
enc_optimizer = optim.SGD(encoder.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(enc_optimizer, step_size=SGD_STEP, gamma=SGD_GAMMA)

# directory name to save the models
MODEL_SAVE_PATH = os.path.join(args.model_save_path, 'Results', 'Model', 'slowfast')
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
SAMPLES_PATH = os.path.join(args.model_save_path, 'Results', 'Samples', 'slowfast')
if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)
LOG_FILE_NAME = 'log.txt'
logging.basicConfig(filename=os.path.join(MODEL_SAVE_PATH, LOG_FILE_NAME),
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
# Load the pretrained model
# The backbones are pre-trained on ImageNet
if PRE_TRAIN is not None:
    pretrained_dict = torch.load(PRE_TRAIN, map_location='cpu')
    try:
        model_dict = encoder.module.state_dict()
    except AttributeError:
        model_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print("load pretrain model")
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)

# check with Yueming how she did it to embed gpu usage
# model = model.cuda(use_gpu[0])
# model = nn.DataParallel(model, device_ids=use_gpu)  # multi-Gpu
INPUT_DIM = 512
OUTPUT_DIM = [(3, 128, 256), (1, 128, 256)]
list_decoders = []
list_optimisers = []
for i in range(2):
    dec = Decoder(OUTPUT_DIM[i], INPUT_DIM, L).to(device)
    opt = optim.Adam(dec.parameters(), lr=LR)
    list_decoders.append(dec)
    list_optimisers.append(opt)

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')
train_batch = len(train_dataloader)
test_batch = len(test_dataloader)


T = config['temp']
avg_cost = np.zeros([EPOCHS, 12], dtype=np.float32)
lambda_weight = np.ones([2, EPOCHS])

for epoch in range(EPOCHS):
    cost = np.zeros(12, dtype=np.float32)
    encoder.train()
    conf_mat = ConfMatrix(encoder.nb_classes)
    end = time.time()
    for step, (inputs, labels, depth) in enumerate(train_dataloader):
        # data_time.update(time.time() - end)
        # [1, 3, 128, 256] = 1, 3, 32768
        inputs = inputs.to(device)
        # split the video into keyframes and non-keyframes
        # torch.Size([1, 128, 256])
        labels = labels.long().to(device)
        # torch.Size([1, 1, 128, 256])
        depth = depth.to(device)
        slow_encoded_output, fast_encoded_output = encoder(inputs)
        output_segmention = torch.zeros(3, labels.shape[0], labels.shape[1])
        output_depth = torch.zeros(1, labels.shape[0], labels.shape[1])
        task_output = [output_segmention, output_depth].to(device)
        se_block_fts = []
        for t in range(NUMBER_CLASSES):
            decoder = list_decoders[t]
            task_output[t], se_block_fts[t] = decoder(prev_sblock_kf, current_se_block_outputs, t)

        enc_optimizer.zero_grad()
        # loss = criterion(outputs, labels)
        loss = [model_fit(pred_output[0], labels, 'semantic'),
                      model_fit(pred_output[1], depth, 'depth')]

        # if config['weight'] == 'equal' or config['weight'] == 'dwa':
        loss = sum([lambda_weight[i, epoch] * loss[i] for i in range(2)])
        loss.backward()
        enc_optimizer.step()

        # accumulate label prediction for every pixel in training images
        conf_mat.update(pred_output[0].argmax(1).flatten(), labels.flatten())

        cost[0] = loss[0].item()
        cost[3] = loss[1].item()
        cost[4], cost[5] = depth_error(pred_output[1], depth)
        avg_cost[epoch, :6] += cost[:6] / train_batch

    # compute mIoU and acc
    avg_cost[epoch, 1:3] = conf_mat.get_metrics()

    # evaluating test data
    model.eval()
    conf_mat = ConfMatrix(model.nb_classes)
    with torch.no_grad():  # operations inside don't track history
        for step, (inputs, labels, depth) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            depth = depth.to(device)
            pred_output = model(inputs)
            val_loss = [model_fit(pred_output[0], labels, 'semantic'),
                         model_fit(pred_output[1], depth, 'depth')]
            conf_mat.update(pred_output[0].argmax(1).flatten(), labels.flatten())
            cost[6] = val_loss[0].item()
            cost[9] = val_loss[1].item()
            cost[10], cost[11] = depth_error(pred_output[1], depth)
            avg_cost[epoch, 6:] += cost[6:] / test_batch

        # compute mIoU and acc
        avg_cost[epoch, 7:9] = conf_mat.get_metrics()

    scheduler.step()
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
          .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
                  avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7],
                  avg_cost[epoch, 8],
                  avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11]))