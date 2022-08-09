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
from utils.metrics import SegmentationMetrics, ConfMatrix, depth_error
from loss.loss import model_fit, DiceLoss, DiceBCELoss
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

def log_scalar(name, value, step, writer):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


def mutli_task_trainer(epoch, train_loader, model, enc_optimizer_slow, enc_optimizer_fast, decocder_opt, writer):
    total_epoch = 10
    cost = np.zeros(7, dtype=np.float32)
    avg_cost = [0, 0, 0, 0]
    epoch_loss_model = 0
    epoch_pixel_accuracy_train = 0
    model.train()
    conf_mat = ConfMatrix(19)
    # end = time.time()
    lambda_weight = np.ones([3, total_epoch])
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    for batch_idx, (inputs, labels, depth) in enumerate(train_loader):
        # data_time.update(time.time() - end)
        # [1, 3, 128, 256] = 1, 3, 32768
        inputs = inputs.to(device)
        # torch.Size([1, 128, 256])
        gt_semantic_labels = labels.long().to(device)
        # torch.Size([1, 1, 128, 256])
        gt_depth = depth.to(device)
        task_pred = model(inputs)

        enc_optimizer_slow.zero_grad()
        enc_optimizer_fast.zero_grad()
        decocder_opt.zero_grad()
        # loss = criterion(semantic_predictions, semantic_labels, ignore_index=-1)
        train_loss = [model_fit(task_pred[0], gt_semantic_labels, 'semantic'),
                      model_fit(task_pred[1], gt_depth, 'depth')]
        # if config['weight'] == 'equal' or config['weight'] == 'dwa':
        loss = sum([lambda_weight[i, epoch] * train_loss[i] for i in range(3)])
        loss.backward()
        enc_optimizer_slow.step()
        enc_optimizer_fast.step()
        decocder_opt.step()
        cost[0] = loss[0].item()
        cost[3] = loss[1].item()
        cost[4], cost[5] = depth_error(task_pred[1], gt_depth)
        # cost[6] = train_loss[2].item()
        # cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
        avg_cost[0] += cost[:6] / len(train_loader)
        if batch_idx % 50 == 0:
            print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\t Segmentation Loss: {:.6f} Depth Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss[0].data.item(),
                    loss[1].data.item()
                )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar("Semantic Segmentation Loss", loss[0].data.item(), step, writer)
            log_scalar("Depth Estimation Loss", loss[1].data.item(), step, writer)
            model.log_weights(step)
        # accumulate label prediction for every pixel in training images
        # conf_mat.update(pred_output[0].argmax(1).flatten(), labels.flatten())
        epoch_loss_model += loss.item()
        pixel_accuracy, dice, precision, recall = metric_calculator(gt_semantic_labels, task_pred[0])
        epoch_pixel_accuracy_train += pixel_accuracy
    avg_cost[1:3] = np.array(conf_mat.get_metrics())
    # returns the average loss per decoder and the loss of the encoder per epoch
    epoch_train_loss = epoch_loss_model / len(train_loader)
    epoch_pixel_accuracy_train = epoch_pixel_accuracy_train / len(train_loader)
    return epoch_train_loss, epoch_pixel_accuracy_train, avg_cost


def test_multi_task(epoch, test_loader, model, criterion, writer, scheduler_slow, scheduler_fast, SAMPLES_PATH):
    # evaluating test data
    epoch_loss_model = 0
    epoch_pixel_accuracy_test = 0
    model.eval()
    conf_mat = ConfMatrix(model.nb_classes)
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    with torch.no_grad():  # operations inside don't track history
        for batch_idx, (inputs, labels, depth) in enumerate(test_loader):
            inputs = inputs.to(device)
            semantic_labels = labels.long().to(device)
            depth = depth.to(device)
            semantic_predictions = model(inputs)
            loss = criterion(semantic_predictions, semantic_labels, ignore_index=-1)
            # val_loss = [model_fit(pred_output[0], labels, 'semantic'),
            #             model_fit(pred_output[1], depth, 'depth')]
            # conf_mat.update(pred_output[0].argmax(1).flatten(), labels.flatten())
            # cost[6] = val_loss[0].item()
            # cost[9] = val_loss[1].item()
            # cost[10], cost[11] = depth_error(pred_output[1], depth)
            # avg_cost[epoch, 6:] += cost[6:] / test_batch
            if batch_idx % 50 == 0:
                pixel_accuracy, dice, precision, recall = metric_calculator(semantic_labels, semantic_predictions)
                epoch_pixel_accuracy_test += pixel_accuracy
                print(
                        "Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss.data.item(),
                    )
                )
                step = epoch * len(test_loader) + batch_idx
                log_scalar("test_loss", loss.data.item(), step, writer)
                model.log_weights(step)
        # compute mIoU and acc
        # avg_cost[epoch, 7:9] = conf_mat.get_metrics()
    # print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
    #       'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
    #       .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
    #               avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7],
    #               avg_cost[epoch, 8],
    #               avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11]))
    # # compute mIoU and acc
    # avg_cost[epoch, 1:3] = conf_mat.get_metrics()
    scheduler_slow.step()
    scheduler_fast.step()
    epoch_test_loss = epoch_loss_model / len(test_loader)
    epoch_pixel_accuracy_test = epoch_pixel_accuracy_test / len(test_loader)
    return epoch_test_loss, epoch_pixel_accuracy_test

def single_task_trainer(epoch, train_loader, model, enc_optimizer_slow, enc_optimizer_fast, decocder_opt, task, writer):
    cost = np.zeros(6, dtype=np.float32)
    avg_cost = np.zeros([1, 6], dtype=np.float32)
    epoch_loss_model = 0
    model.train()
    conf_mat = ConfMatrix(19)
    # end = time.time()
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    for batch_idx, (inputs, labels, depth) in enumerate(train_loader):
        # data_time.update(time.time() - end)
        # [1, 3, 128, 256] = 1, 3, 32768
        inputs = inputs.to(device)
        # torch.Size([1, 128, 256])
        gt_semantic_labels = labels.long().to(device)
        # torch.Size([1, 1, 128, 256])
        gt_depth = depth.to(device)
        # outputs a single task prediction
        task_pred = model(inputs)
        enc_optimizer_slow.zero_grad()
        enc_optimizer_fast.zero_grad()
        decocder_opt.zero_grad()

        if task == 'semantic':
            task_pred = torch.argmax(task_pred, dim=1)
            loss = model_fit(task_pred, gt_semantic_labels, task)
            loss.backward()
            # accumulate label prediction for every pixel in training images
            conf_mat.update(task_pred.argmax(1).flatten(), gt_semantic_labels.flatten())
            cost[0] = loss.item()
            avg_cost[1:3] = np.array(conf_mat.get_metrics())

        if task == 'depth':
            loss = model_fit(task_pred, gt_depth, task)
            loss.backward()
            cost[3] = loss.item()
            cost[4], cost[5] = depth_error(task_pred, gt_depth)

        # if task == 'normal':
        #     train_loss = model_fit(task_pred, train_normal, task)
        #     train_loss.backward()
        #     cost[6] = train_loss.item()
        #     cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

        enc_optimizer_slow.step()
        enc_optimizer_fast.step()
        decocder_opt.step()
        epoch_loss_model += loss.item()
        avg_cost[:6] += cost[:6] / len(train_loader)
        if batch_idx % 50 == 0:
            print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\t {} Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    task,
                    loss.data.item()
                    )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar(task+" Loss", loss.data.item(), step, writer)
            model.log_weights(step)

    # returns the average loss per decoder and the loss of the encoder per epoch
    epoch_train_loss = epoch_loss_model / len(train_loader)
    return epoch_train_loss, cost, avg_cost

def test_single_task(epoch, test_loader, single_task_model, writer, scheduler_slow, scheduler_fast, task, SAMPLES_PATH):
    # evaluating test data
    cost = np.zeros(6, dtype=np.float32)
    avg_cost = np.zeros([1, 6], dtype=np.float32)
    epoch_loss_model = 0
    # epoch_pixel_accuracy_test = 0
    single_task_model.eval()
    conf_mat = ConfMatrix(single_task_model.nb_classes)
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    with torch.no_grad():
        for batch_idx, (inputs, labels, depth) in enumerate(test_loader):
            inputs = inputs.to(device)
            gt_semantic_labels = labels.long().to(device)
            gt_depth = depth.to(device)
            task_pred = single_task_model(inputs)

            if task == 'semantic':
                loss = model_fit(task_pred, gt_semantic_labels, task)
                loss.backward()
                # accumulate label prediction for every pixel in training images
                conf_mat.update(task_pred.argmax(1).flatten(), gt_semantic_labels.flatten())
                cost[0] = loss.item()
                avg_cost[1:3] = np.array(conf_mat.get_metrics())

            if task == 'depth':
                loss = model_fit(task_pred, gt_depth, task)
                loss.backward()
                cost[3] = loss.item()
                cost[4], cost[5] = depth_error(task_pred, gt_depth)

            # if task == 'normal':
            #     train_loss = model_fit(task_pred, train_normal, task)
            #     train_loss.backward()
            #     cost[6] = train_loss.item()
            #     cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

            avg_cost[:6] += cost[:6] / len(test_loader)
            if batch_idx % 50 == 0:
                # pixel_accuracy, dice, precision, recall = metric_calculator(task_pred, gt_semantic_labels)
                # epoch_pixel_accuracy_test += pixel_accuracy
                print(
                        "Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss.data.item(),
                    )
                )
                step = epoch * len(test_loader) + batch_idx
                log_scalar("test_loss", loss.data.item(), step, writer)
                single_task_model.log_weights(step)
        # compute mIoU and acc
        # avg_cost[epoch, 7:9] = conf_mat.get_metrics()
    # print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
    #       'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
    #       .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
    #               avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7],
    #               avg_cost[epoch, 8],
    #               avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11]))
    # # compute mIoU and acc
    # avg_cost[epoch, 1:3] = conf_mat.get_metrics()
    scheduler_slow.step()
    scheduler_fast.step()
    epoch_test_loss = epoch_loss_model / len(test_loader)
    return epoch_test_loss, cost, avg_cost


def static_single_task_trainer(epoch, train_loader, model, enc_optimizer, decoder_opt, task, writer):
    cost = np.zeros(6, dtype=np.float32)
    avg_cost = np.zeros([1, 6], dtype=np.float32)
    epoch_loss_model = 0
    model.train()
    conf_mat = ConfMatrix(19)
    # end = time.time()
    loss = float('inf')
    # initialise the loss the function
    criterion = DiceBCELoss()
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    for batch_idx, (inputs, labels, depth) in enumerate(train_loader):
        # data_time.update(time.time() - end)
        # [8, 1, 256, 512]
        inputs = inputs.float().to(device)
        # torch.Size([8, 1, 256, 512])
        gt_semantic_labels = labels.float().to(device)
        # torch.Size([8, 3, 256, 512])
        gt_depth = depth.to(device)
        # outputs a single task prediction
        task_pred = model(inputs)
        # task_pred.mean().backward()  # works
        print(task_pred.grad_fn)
        print(task_pred.shape)
        enc_optimizer.zero_grad()
        decoder_opt.zero_grad()

        if task == 'segmentation':
            gt_semantic_labels = torch.argmax(gt_semantic_labels, dim=1)
            task_pred = torch.argmax(task_pred, dim=1)

            print(gt_semantic_labels.shape)
            print(task_pred.shape)
            loss = criterion(task_pred, gt_semantic_labels)
            # loss = model_fit(task_pred, gt_semantic_labels, task)
            loss.backward()
            # accumulate label prediction for every pixel in training images
            conf_mat.update(task_pred.argmax(1).flatten(), gt_semantic_labels.flatten())
            cost[0] = loss.item()
            avg_cost[1:3] = np.array(conf_mat.get_metrics())

        if task == 'depth':
            loss = model_fit(task_pred, gt_depth, task)
            loss.backward()
            cost[3] = loss.item()
            cost[4], cost[5] = depth_error(task_pred, gt_depth)

        # if task == 'normal':
        #     train_loss = model_fit(task_pred, train_normal, task)
        #     train_loss.backward()
        #     cost[6] = train_loss.item()
        #     cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

        enc_optimizer.step()
        decoder_opt.step()
        epoch_loss_model += loss.item()
        avg_cost[:6] += cost[:6] / len(train_loader)
        if batch_idx % 50 == 0:
            print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\t {} Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    task,
                    loss.data.item()
                    )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar(task+" Loss", loss.data.item(), step, writer)
            model.log_weights(step)

    # returns the average loss per decoder and the loss of the encoder per epoch
    epoch_train_loss = epoch_loss_model / len(train_loader)
    return epoch_train_loss, cost, avg_cost


def static_test_single_task(epoch, test_loader, single_task_model, writer, scheduler, task, SAMPLES_PATH):
    # evaluating test data
    cost = np.zeros(6, dtype=np.float32)
    avg_cost = np.zeros([1, 6], dtype=np.float32)
    epoch_loss_model = 0
    # epoch_pixel_accuracy_test = 0
    single_task_model.eval()
    conf_mat = ConfMatrix(single_task_model.nb_classes)
    metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
    with torch.no_grad():
        for batch_idx, (inputs, labels, depth) in enumerate(test_loader):
            inputs = inputs.to(device)
            gt_semantic_labels = labels.long().to(device)
            gt_depth = depth.to(device)
            task_pred = single_task_model(inputs)

            if task == 'semantic':
                # loss = model_fit(task_pred, gt_semantic_labels, task)
                loss = DiceLoss(task_pred, gt_semantic_labels)
                print(gt_semantic_labels.shape)
                loss.backward(task_pred, gt_semantic_labels)
                # accumulate label prediction for every pixel in training images
                conf_mat.update(task_pred.argmax(1).flatten(), gt_semantic_labels.flatten())
                cost[0] = loss.item()
                avg_cost[1:3] = np.array(conf_mat.get_metrics())

            if task == 'depth':
                loss = model_fit(task_pred, gt_depth, task)
                loss.backward()
                cost[3] = loss.item()
                cost[4], cost[5] = depth_error(task_pred, gt_depth)

            # if task == 'normal':
            #     train_loss = model_fit(task_pred, train_normal, task)
            #     train_loss.backward()
            #     cost[6] = train_loss.item()
            #     cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred, train_normal)

            avg_cost[:6] += cost[:6] / len(test_loader)
            if batch_idx % 50 == 0:
                # pixel_accuracy, dice, precision, recall = metric_calculator(task_pred, gt_semantic_labels)
                # epoch_pixel_accuracy_test += pixel_accuracy
                print(
                        "Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss.data.item(),
                    )
                )
                step = epoch * len(test_loader) + batch_idx
                log_scalar("test_loss", loss.data.item(), step, writer)
                single_task_model.log_weights(step)
        # compute mIoU and acc
        # avg_cost[epoch, 7:9] = conf_mat.get_metrics()
    # print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
    #       'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
    #       .format(epoch, avg_cost[epoch, 0], avg_cost[epoch, 1], avg_cost[epoch, 2], avg_cost[epoch, 3],
    #               avg_cost[epoch, 4], avg_cost[epoch, 5], avg_cost[epoch, 6], avg_cost[epoch, 7],
    #               avg_cost[epoch, 8],
    #               avg_cost[epoch, 9], avg_cost[epoch, 10], avg_cost[epoch, 11]))
    # # compute mIoU and acc
    # avg_cost[epoch, 1:3] = conf_mat.get_metrics()
    scheduler_slow.step()
    scheduler_fast.step()
    epoch_test_loss = epoch_loss_model / len(test_loader)
    return epoch_test_loss, cost, avg_cost








