import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.utils.data.sampler as sampler
import os
import random
import time
import mlflow
from tensorboardX import SummaryWriter
import mlflow.pytorch
from utils.metrics import SegmentationMetrics, ConfMatrix, depth_error, \
    AverageMeter, ProgressMeter, iouCalc, visim, vislbl
from loss.loss import model_fit, DiceLoss, DiceBCELoss
import numpy as np
import logging
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################## STATIC TRAINING/EVALUTATION ########################################

def static_single_task_trainer(epoch, criterion, train_loader, model, model_opt, scheduler, task, LOG_FILE):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    abs_error_running = AverageMeter('Absolute error', ':.3f')
    rel_error_running = AverageMeter('Relative error', ':.3f')

    if task == 'segmentation':
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, loss_running, acc_running],
            prefix="Train, epoch: [{}]".format(epoch))
    if task == 'depth':
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, loss_running, abs_error_running, rel_error_running],
            prefix="Train, epoch: [{}]".format(epoch))
    if task == 'segmentation_depth':
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, loss_running, acc_running, abs_error_running, rel_error_running],
            prefix="Train, epoch: [{}]".format(epoch))

    model.train()
    # conf_mat = ConfMatrix(19)
    # initialise the loss the function
    # metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    end = time.time()
    for batch_idx, (inputs, labels, depth) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # [8, 1, 256, 512]
        model_opt.zero_grad()
        inputs = inputs.float().to(device)
        # torch.Size([8, 1, 256, 512])
        gt_semantic_labels = labels.long().to(device)
        # torch.Size([8, 3, 256, 512])
        gt_depth = depth.to(device)

        # outputs a single task prediction
        if task == 'segmentation_depth':
            depth_pred, seg_pred = model(inputs)
            seg_loss = criterion[0](seg_pred, gt_semantic_labels.squeeze().long())
            depth_loss = criterion[1](seg_pred, gt_depth)

            # Equal Weighted losses
            depth_weight = 0.5
            seg_weight = 0.5
            total_loss = (depth_weight * depth_loss) + (seg_loss * seg_weight)
            total_loss.backward()
            model_opt.step()

            # store total loss
            bs = inputs.size(0)
            total_loss = total_loss.item()
            loss_running.update(total_loss, bs)

            # get segmentation metric
            seg_pred = torch.argmax(seg_pred, dim=1)
            corrects = torch.sum(seg_pred == gt_semantic_labels.data)
            void = 0
            nvoid = int((gt_semantic_labels == void).sum())
            res = 512 * 256
            acc = corrects.double() / (bs * res - nvoid)  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # get depth metric
            abs_err, rel_err = depth_error(depth_pred, gt_depth)
            abs_error_running.update(abs_err)
            rel_error_running.update(rel_err)

        if task == 'segmentation':
            task_pred = model(inputs)
            loss = criterion(task_pred, gt_semantic_labels.squeeze().long())
            # backward pass
            loss.backward()
            model_opt.step()

            bs = inputs.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            task_pred = torch.argmax(task_pred, dim=1)
            corrects = torch.sum(task_pred == gt_semantic_labels.data)
            void = 0
            nvoid = int((gt_semantic_labels == void).sum())
            res = 512*256
            acc = corrects.double() / (bs * res - nvoid)  # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)

            # accumulate label prediction for every pixel in training images
            # conf_mat.update(task_pred.argmax(1).flatten(), gt_semantic_labels.flatten())
            # cost[0] = loss.item()
            # print(len(np.array(conf_mat.get_metrics())))
            # avg_cost[1:3] = np.array(conf_mat.get_metrics())

        if task == 'depth':
            task_pred = model(inputs)
            loss = criterion(task_pred, gt_depth)
            # backward pass
            loss.backward()
            model_opt.step()
            bs = inputs.size(0)  # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            abs_err, rel_err = depth_error(task_pred, gt_depth)
            abs_error_running.update(abs_err)
            rel_error_running.update(rel_err)
        # if task == 'normal':

        # output training info
        progress.display(batch_idx)
        if batch_idx % 25 == 0:
            if task == 'segmentation':
                with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as batch_log:
                    batch_log.write('{}, {:.5f}, {:.5f}, {:.5f}\n'.format(epoch, batch_idx, loss, acc))
            if task == 'depth':
                with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as batch_log:
                    batch_log.write('{}, {}, {:.5f}, {:.5f}, {:.5f}\n'.format(epoch, batch_idx, loss,
                                                                                   abs_err, rel_err))

        # Measure time
        batch_time.update(time.time() - end)
        end = time.time()

        # epoch_loss_model += loss.item()
        # avg_cost[:6] += cost[:6] / len(train_loader)
    # reduce the learning rate
    scheduler.step(loss_running.avg)
    # returns the average loss per decoder and the loss of the encoder per epoch
    # epoch_train_loss = epoch_loss_model / len(train_loader)
    # return epoch_train_loss, cost, avg_cost
    if task == 'depth':
        return loss_running.avg, abs_error_running.avg, rel_error_running.avg

    return loss_running.avg, acc_running.avg


def static_test_single_task(epoch, criterion, test_loader, single_task_model, task, classLabels, validClasses,
                            folder, void=0, maskColors=None, args=None):
    # evaluating test data
    # SAMPLES_PATH
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.4e')
    abs_error_running = AverageMeter('Absolute error', ':.3f')
    rel_error_running = AverageMeter('Relative error', ':.3f')

    if task == 'segmentation':
        iou = iouCalc(classLabels, validClasses, voidClass=void)
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, data_time, loss_running, acc_running],
            prefix="Train, epoch: [{}]".format(epoch))

    if task == 'depth':
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, data_time, loss_running, abs_error_running, rel_error_running],
            prefix="Train, epoch: [{}]".format(epoch))

    if task == 'segmentation_depth':
        iou = iouCalc(classLabels, validClasses, voidClass=void)
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, data_time, loss_running, acc_running, abs_error_running, rel_error_running],
            prefix="Train, epoch: [{}]".format(epoch))

    single_task_model.eval()
    # conf_mat = ConfMatrix(single_task_model.nb_classes)
    # metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, labels, depth, filepath) in enumerate(test_loader):
            inputs = inputs.float().to(device)
            gt_semantic_labels = labels.long().to(device)
            gt_depth = depth.to(device)
            task_pred = single_task_model(inputs)

            if task == 'segmentation':
                loss = criterion(task_pred, gt_semantic_labels.squeeze().long())
                bs = inputs.size(0)
                loss = loss.item()
                loss_running.update(loss, bs)
                task_pred = torch.argmax(gt_semantic_labels, 1)
                corrects = torch.sum(task_pred == gt_semantic_labels.data)
                void = 0
                nvoid = int((gt_semantic_labels == void).sum())
                res = 512 * 256
                acc = corrects.double() / (bs * res - nvoid)  # correct/(batch_size*resolution-voids)
                acc_running.update(acc, bs)
                # print(task_pred.squeeze().shape)
                # print(gt_semantic_labels.squeeze().shape)
                # torch.set_printoptions(profile="full")
                # t = task_pred.squeeze()
                # print(t[0, :, :])
                # b = gt_semantic_labels.squeeze()
                # print(b[0,:,:])
                # torch.set_printoptions(profile="default")
                # Calculate IoU scores of current batch
                # iou.evaluateBatch(task_pred.squeeze(), gt_semantic_labels.squeeze())
                # iou.evaluateBatch(decode_pred(task_pred.squeeze(), validClasses), gt_semantic_labels.squeeze())

                # Save visualizations of first batch
                if batch_idx == 0 and maskColors is not None:
                    for i in range(inputs.size(0)):
                        filename = filepath[i]
                        # Only save inputs and labels once
                        if epoch == 0:
                            img = visim(inputs[i, :, :, :], args)
                            label = vislbl(labels[i, :, :], maskColors)
                            if len(img.shape) == 3:
                                cv2.imwrite(folder + '/images/{}.png'.format(filename), img[:, :, ::-1])
                            else:
                                cv2.imwrite(folder + '/images/{}.png'.format(filename), img)
                            cv2.imwrite(folder + '/images/{}_gt.png'.format(filename), label[:, :, ::-1])
                        # Save predictions
                        pred = vislbl(preds[i, :, :], maskColors)
                        cv2.imwrite(folder + '/images/{}_epoch_{}.png'.format(filename, epoch), pred[:, :, ::-1])

            if task == 'depth':
                loss = criterion(task_pred, gt_depth)
                bs = inputs.size(0)  # current batch size
                loss = loss.item()
                loss_running.update(loss, bs)
                abs_err, rel_err = depth_error(task_pred, gt_depth)
                abs_error_running.update(abs_err)
                rel_error_running.update(rel_err)

        # compute mIoU and acc
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress info
        progress.display(epoch)
    # # compute mIoU and acc
    # avg_cost[epoch, 1:3] = conf_mat.get_metrics()
    # scheduler_slow.step()
    # scheduler_fast.step()
    # epoch_test_loss = epoch_loss_model / len(test_loader)
    # return epoch_test_loss, cost, avg_cost
    if task == 'depth':
        return rel_error_running.avg, abs_error_running.avg, loss_running.avg

    # miou = iou.outputScores()
    print('Accuracy      : {:5.3f}'.format(acc_running.avg))
    print('---------------------')
    return acc_running.avg, loss_running.avg, miou

# def decode_pred(input, validClasses):
#     # Put all void classes to zero
#     for i in range(input.size(0)):
#         for _predc in range(len(validClasses)):
#             input[i, input == _predc, input == _predc] = validClasses[_predc]
#     return input


