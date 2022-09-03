import numpy as np
import torch
import argparse
# import itertools
import tempfile
import oyaml as yaml
import logging
# from tensorboardX import SummaryWriter
# from torchvision.datasets import Cityscapes
from loader.cityscapes_loader import cityscapesLoader
from loader.static_loader import staticLoader
from loader.temporal_loader import temporalLoader
# from loader.video_dataset import *
from utils.sort_dataset import *
# from loader.nyuv2_dataloader import NYUV2
from models.decoder import SegDecoder, DepthDecoder, MultiDecoder, DecoderTemporal
from models.mtl_model import TemporalModel
from models.single_backbone_temporal import TemporalModel2
from models.static_model import StaticTaskModel
from models.deeplabv3_encoder import DeepLabv3
from utils.train_helpers import *
# from utils.static_helpers import static_single_task_trainer, static_test_single_task, save_ckpt
from utils.temporal_helpers import static_single_task_trainer, static_test_single_task, save_ckpt

from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback, SynchronizedBatchNorm3d
from utils.metrics import plot_learning_curves
from loss.loss import InverseDepthL1Loss, L1LossIgnoredRegion, consistency_weight,  \
    softmax_mse_loss, softmax_kl_loss, softmax_js_loss
# from scheduler import get_scheduler
# from deeplabv3plus import Deeplab_v3plus
# import deeplab
from utils.metrics_seg import StreamSegMetrics
from utils import ext_transforms as et

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

# commandline arguments
parser = argparse.ArgumentParser(description='Multi-Task Temporal w/ full annotation')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('--model_save_path', type=str, default='.', help='Path to the saved models')
# parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
# parser.add_argument('--window_size', type=int, default=5, help='ILA module computes on a window size')
parser.add_argument('--dataset_mean', metavar='[0.485, 0.456, 0.406]',
                    default=[0.485, 0.456, 0.406], type=list,
                    help='mean for normalization')
parser.add_argument('--dataset_std', metavar='[0.229, 0.224, 0.225]',
                    default=[0.229, 0.224, 0.225], type=list,
                    help='std for normalization')
parser.add_argument('-unsup', '--unsup', default=False, type=bool,
                    help='Bool declaring add unsupervised pathway to network')
parser.add_argument('-semisup', '--semisup', default=False, type=bool,
                    help='Bool declaring add semi supervision pathway to network')
parser.add_argument('-v', '--version', default='sum_fusion', type=str,
                    help='Adding the fusion method')
parser.add_argument('-c', '--causal', default=True, type=bool,
                    help='Checking the causal pathway')
# uncomment for segmentation run
# parser.add_argument("--config", default='configs/medtronic_cluster/temporal_cityscape_config_seg',
#                     nargs="?", type=str, help="Configuration file to use")

# uncomment for depth run
# parser.add_argument("--config", default='configs/medtronic_cluster/temporal_cityscape_config_depth',
#                     nargs="?", type=str, help="Configuration file to use")

# uncomment for both tasks
parser.add_argument("--config", default='configs/medtronic_cluster/temporal_cityscape_config_both',
                    nargs="?", type=str, help="Configuration file to use")

args = parser.parse_args()
with open(args.config) as fp:
    cfg = yaml.safe_load(fp)

# set seed for reproducibility
init_seed(12345, en_cudnn=False)
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('number of gpu   : {:6d}'.format(num_gpu))

EPOCHS = cfg["training"]["train_iters"]
BACKBONE = cfg["model"]["backbone"]["encoder"]
DATASET_PATH = cfg["data"]["path"]
CLASS_TASKS = cfg["model"]["task_classes"]
frames_per_segment = cfg["model"]["frames_per_segment"]
window_size = cfg["model"]["window_size"]
k = cfg["model"]["k"]
data_path = cfg["data"]["path"]
input_dim_decoder = 256
TASK = cfg["model"]["tasks"]
# version = cfg["model"]["version"]
version = args.version
unsup_ = args.unsup
semi_sup_ = args.semisup
causal = args.causal

print('Running experiment on Task: ' + TASK)
print('Temporal version: ' + version)
print('Adding unsupervised learning to encoders: ' + str(unsup_))
print('Adding semi supervision to network: ' + str(semi_sup_))

# Initialize mlflow
NAME_EXPERIMENT = 'experiment_temporal_' + TASK
mlflow.set_experiment(experiment_name=NAME_EXPERIMENT)

train_transform = et.ExtCompose([
            et.ExtResize((128, 256)),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            # et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

val_transform = et.ExtCompose([
        et.ExtResize((128, 256)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

# Load the dataloaders
print('Preprocessing Temporal ' + cfg["data"]["dataset"])
train_set = temporalLoader(DATASET_PATH,
                           split=cfg["data"]["train_split"],
                           transform=train_transform,
                           test_mode=False,
                           model_name=None,
                           interval=window_size)
val_set = temporalLoader(DATASET_PATH,
                         split=cfg["data"]["val_split"],
                         transform=val_transform,
                         test_mode=True,
                         model_name=None,
                         interval=window_size)
test_set = temporalLoader(DATASET_PATH,
                         split=cfg["data"]["test_split"],
                         transform=val_transform,
                         test_mode=True,
                         model_name=None,
                         interval=window_size)

classLabels = val_set.class_names
print(len(classLabels))
validClasses = val_set.valid_classes
# mask_colors = val_set.colors
void_class = val_set.void_classes

train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=cfg["training"]["batch_size"],
                                               shuffle=False,
                                               num_workers=cfg["training"]["n_workers"],
                                               drop_last=True)

val_dataloader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=cfg["validating"]["batch_size"],
                                             shuffle=False,
                                             num_workers=cfg["validating"]["n_workers"],
                                             drop_last=True)

test_dataloader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=cfg["validating"]["batch_size"],
                                              shuffle=False,
                                              num_workers=cfg["validating"]["n_workers"],
                                              drop_last=True)
print('Initialising Temporal Model')
# Initializing encoder
deeplabv3_backbone_slow = cfg["model"]["backbone"]["encoder"]["resnet_slow"]
deeplabv3_backbone_fast = cfg["model"]["backbone"]["encoder"]["resnet_fast"]

# encoder_slow = DeepLabv3(deeplabv3_backbone_slow).to(device)
# encoder_fast = DeepLabv3(deeplabv3_backbone_fast).to(device)

# enc_optimizer = optim.SGD(enc.parameters(),
#                           lr=cfg["training"]["optimizer"]["lr0"],
#                           momentum=cfg["training"]["optimizer"]["momentum"],
#                           weight_decay=cfg["training"]["optimizer"]["wd"])

# initialise decoders (one for each task)
drop_out = cfg["model"]["dropout"]
# version = 'sum_fusion'
# version = 'convnet_fusion'
# version = 'global_atten_fusion'
# version = 'conv3d_fusion'
# version = 'sum_fusion'
# version = 'causal_fusion'
# unsup_ = False
# semi_sup_ = False
# mulit_task_ = False

window_size = 3 

# if we are to add semi-supervision on the unlabelled frames
if semi_sup_:
    ram_up = 0.1
    unsupervised_w = 30
    rampup_ends = int(ram_up * EPOCHS)
    cons_w_unsup = consistency_weight(final_w=unsupervised_w, iters_per_epoch=len(train_dataloader),
                                      rampup_ends=rampup_ends)
    semisup_loss = {'Mode': True, 'Function': softmax_mse_loss, 'Weights': cons_w_unsup, 'Epochs': EPOCHS}
else:
    semisup_loss = {'Mode': False, 'Function': None}

if unsup_:
    unsup_loss = {'Mode': True, 'Function': torch.nn.L1Loss()}
else:
    unsup_loss = {'Mode': False, 'Function': None}

if TASK == 'depth_segmentation':
    mulit_task_ = True
else:
    mulit_task_ = False

# initialise multi (or single) - task model
# model = TemporalModel(cfg, TASK, CLASS_TASKS, drop_out,
#                       window_size, k, semisup_loss, unsup_loss, version=version, mulit_task=mulit_task_).to(device)

model = TemporalModel2(cfg, TASK, CLASS_TASKS, drop_out,
                       window_size, k, semisup_loss, unsup_loss,
                       version=version, mulit_task=mulit_task_, causual_first_layer=causal).to(device)

# model = model.to(device)
# Push model to GPU
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).to(device)
    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

model_opt = optim.Adam(model.parameters(), lr=0.0001)
# model_opt = optim.Adam(model.parameters(), lr=cfg["training"]["optimizer"]["lr0"])
scheduler = optim.lr_scheduler.StepLR(model_opt,
                                      step_size=100,
                                      gamma=0.5)

# directory name to save the models
if semi_sup_ and  unsup_:
    file = TASK + '_semisup_and_unsup_'
elif semi_sup_ and  not unsup_:
    file = TASK + '_semisup_'
elif not semi_sup_ and unsup_:
    file = TASK + '_unsup_'
elif causal:
    file = TASK + '_causal_slow_fast'
else:
    file = TASK

MODEL_SAVE_PATH = os.path.join(args.model_save_path, 'Model', 'Checkpoints', 'Temporal', version, file)
LOG_FILE = os.path.join(args.model_save_path, 'Logs', 'Temporal', version, file)
SAMPLES_PATH = os.path.join(args.model_save_path, 'Model', 'Sample', 'Temporal', version, file)

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)

# Generate log file
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)

# for each training batch record important metrics
if TASK == 'segmentation':
    # for each epoch record important metrics
    with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
        epoch_log.write('epoch, loss, Pix. acc, mIoU \n')

    with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as f:
        f.write('Epoch, Batch Index, train loss, train acc\n')

if TASK == 'depth':
    # for each epoch record important metrics
    with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
        epoch_log.write('epoch, loss, Abs. error \n')

    with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as f:
        f.write('Epoch, Batch Index, train loss, abs_err, rel_err\n')

if TASK == 'depth_segmentation':
    # for each epoch record important metrics
    with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
        epoch_log.write('epoch, loss, Abs. error, Pix. acc, mIoU \n')

    with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as f:
        f.write('Epoch, Batch Index, train loss, train acc, train abs error, train rel error \n')

LOG_FILE_NAME = 'log.txt'
start_epoch = 0
# Load the pretrained model
if cfg['training']['resume'] is not None:
    if os.path.isfile(cfg['training']['resume']):
        checkpoint = torch.load(cfg['training']['resume'])
        # checkpoint = torch.load(cfg['training']['resume'], map_location=lambda storage, loc: storage)  # load model trained on gpu on cpu
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        # start_iter = checkpoint["epoch"]
        print("Loaded checkpoint '{}' (iter {})".format(cfg['training']['resume'], checkpoint["epoch"]))
    else:
        print("No checkpoint found at '{}'".format(cfg['training']['resume']))

# print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')
train_batch = len(train_dataloader)
test_batch = len(test_dataloader)

since = time.time()

if TASK == 'segmentation':
    # Set up metrics
    if not os.path.exists(os.path.join(SAMPLES_PATH, 'images')):
        os.makedirs(os.path.join(SAMPLES_PATH, 'images'))
    # Initialize metrics
    best_miou = 0.0
    metrics = {'train_loss': [],
               'train_acc': [],
               'train_miou': [],
               'val_acc': [],
               'val_loss': [],
               'val_miou': []}
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_set.ignore_class, reduction='mean')
    # criterion = utils.FocalLoss(ignore_index=255, size_average=True)

elif TASK == 'depth':
    if not os.path.exists(os.path.join(SAMPLES_PATH, 'images')):
        os.makedirs(os.path.join(SAMPLES_PATH, 'images'))
    # Initialize metrics
    metrics = {'train_loss': [],
               'val_loss': [],
               'train_abs_error': [],
               'train_rel_error': [],
               'val_abs_error': [],
               'val_rel_error': []}
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    # criterion = InverseDepthL1Loss()
    criterion = L1LossIgnoredRegion()

elif TASK == 'depth_segmentation':
    if not os.path.exists(os.path.join(SAMPLES_PATH, 'images')):
        os.makedirs(os.path.join(SAMPLES_PATH, 'images'))
    # Initialize metrics
    best_miou = 0.0
    metrics = {'train_loss': [],
               'val_loss': [],
               'train_abs_error': [],
               'train_rel_error': [],
               'train_acc': [],
               'train_miou': [],
               'val_abs_error': [],
               'val_rel_error': [],
               'val_acc': [],
               'val_miou': []}
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    # criterion = [InverseDepthL1Loss(), torch.nn.CrossEntropyLoss(ignore_index=train_set.ignore_index)]
    criterion = [L1LossIgnoredRegion(), torch.nn.CrossEntropyLoss(ignore_index=train_set.ignore_class, reduction='mean')]

print('Beginning training...')
with mlflow.start_run():

    # Logging the important parameters for the model
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('task', TASK)
    mlflow.log_param('training_batch_size', cfg["training"]["batch_size"])
    mlflow.log_param('number_of_workers', cfg["training"]["n_workers"])
    mlflow.log_param('backbone', BACKBONE)
    mlflow.log_param('window_size', window_size)
    mlflow.log_param('k', k)
    mlflow.log_param('deeplabv3_backbone_slow', deeplabv3_backbone_slow)
    mlflow.log_param('deeplabv3_backbone_fast', deeplabv3_backbone_fast)
    mlflow.log_param('encoder_pretrained', cfg["training"]["pretrain"])
    mlflow.log_param('dropout', drop_out)
    mlflow.log_param('model_optimizer', cfg["training"]["optimizer"]["name"])
    mlflow.log_param('model_optimizer_lr', cfg["training"]["optimizer"]["lr0"])
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    # # Create a SummaryWriter to write TensorBoard events locally
    # # output_dir = dirpath = tempfile.mkdtemp()
    # output_file = tempfile.mkdtemp()
    # writer = SummaryWriter(output_file)
    # print("Writing TensorBoard events locally to %s\n" % output_file)
    # lambda_weight = np.ones([2, EPOCHS])

    lowest_depth_error = float('inf')
    for epoch in range(start_epoch, EPOCHS):

        print('--- Training ---')
        if TASK == 'segmentation':
            train_loss, train_acc, train_miou = static_single_task_trainer(epoch, criterion, semisup_loss,
                                                                           unsup_loss, train_dataloader, model, model_opt,
                                                                           scheduler, TASK, LOG_FILE)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['train_miou'].append(train_miou)
            print('Epoch {} train loss: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                                  train_loss,
                                                                                  train_acc,
                                                                                  train_miou))

            # Validate
            if epoch % cfg['training']['val_interval'] == 0:
                print('--- Validation ---')
                val_acc, val_loss, val_miou = static_test_single_task(epoch, criterion, semisup_loss, unsup_loss,
                                                                      val_dataloader, model, TASK, SAMPLES_PATH,
                                                                      cfg, save_val_imgs=True)
                metrics['val_acc'].append(val_acc)
                metrics['val_loss'].append(val_loss)
                metrics['val_miou'].append(val_miou)

                print('Validation: epoch {} val loss: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch, val_loss, val_acc, val_miou))
                # Save latest validation checkpoint
                path_save_model = os.path.join(MODEL_SAVE_PATH, 'latest_val_checkpoint.pth.tar')
                save_ckpt(path_save_model, model, model_opt, scheduler, metrics, val_miou, epoch)
                # # Since the model was logged as an artifact, it can be loaded to make predictions
                # mlflow.pytorch.log_model(model, artifact_path="pytorch-" + TASK + "-trained", pickle_module=pickle)
                # print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(),
                #                                                      "pytorch-" + TASK + "-trained"))

                with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
                    epoch_log.write('Validation: {}, {:.5f}, {:.5f}, {:.5f}\n'.format(epoch, val_loss,
                                                                                      val_acc, val_miou))
                # Save best model to file
                if val_miou > best_miou:
                    print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, val_miou))
                    best_miou = val_miou
                    path_save_model = os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth.tar')
                    save_ckpt(path_save_model, model, model_opt, scheduler, metrics, best_miou, epoch)
                    mlflow.log_metric('best_miou', best_miou)
                    # print("\nLogging the trained model as a run artifact...")
                    # mlflow.pytorch.log_model(model, artifact_path="pytorch-" + TASK + "-best", pickle_module=pickle)

            # Write segmentation logs epoch, train loss, val loss, train acc, val acc, miou

            with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
                epoch_log.write('Train: {}, {:.5f}, {:.5f}, {:.5f}\n'.format(epoch, train_loss, train_acc, train_miou))
            mlflow.log_artifact(os.path.join(LOG_FILE, 'log_results.txt'))



        elif TASK == 'depth':
            train_loss, train_abs_err, train_rel_err = static_single_task_trainer(epoch, criterion, semisup_loss,
                                                                                  unsup_loss, train_dataloader, model,
                                                                                  model_opt, scheduler, TASK, LOG_FILE)
            metrics['train_loss'].append(train_loss)
            metrics['train_abs_error'].append(train_abs_err)
            metrics['train_rel_error'].append(train_rel_err)

            print('Epoch {} train loss: {:.4f}, abs error: {:.4f}, val rel error: {:.4f}'.format(epoch,
                                                                                                 train_loss,
                                                                                                 train_abs_err,
                                                                                                 train_rel_err))

            # Validate
            if epoch % cfg['training']['val_interval'] == 0:
                print('--- Validation ---')
                val_rel_error, val_abs_error, val_loss = static_test_single_task(epoch, criterion, semisup_loss, unsup_loss,
                                                                                 val_dataloader, model, TASK, SAMPLES_PATH,
                                                                                 cfg, save_val_imgs=True)
                metrics['val_abs_error'].append(val_abs_error)
                metrics['val_loss'].append(val_loss)
                metrics['val_rel_error'].append(val_rel_error)

                print('Validation: epoch {} val loss: {:.4f}, val abs error: {:.4f}, val rel error: {:.4f}'.format(epoch,
                                                                                                                   val_loss,
                                                                                                                   val_abs_error,
                                                                                                                   val_rel_error))
                # Save latest validation checkpoint
                path_save_model = os.path.join(MODEL_SAVE_PATH, 'latest_val_checkpoint.pth.tar')
                save_ckpt(path_save_model, model, model_opt, scheduler, metrics, val_abs_error, epoch)
                # print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(),
                #                                                      "pytorch-" + TASK + "-trained"))

                with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
                    epoch_log.write('Validation: {}, {:.5f}, {:.5f} \n'.format(epoch, val_loss, val_abs_error))

                # Save best model to file
                if val_abs_error < lowest_depth_error:
                    print('Val error improved from {:.4f} to {:.4f}.'.format(lowest_depth_error, val_abs_error))
                    lowest_depth_error = val_abs_error
                    path_save_model = os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth.tar')
                    save_ckpt(path_save_model, model, model_opt, scheduler, metrics, lowest_depth_error, epoch)
                    mlflow.log_metric('best_abs_error', lowest_depth_error)
                    # mlflow.pytorch.log_model(model, artifact_path="pytorch-" + TASK + "-best", pickle_module=pickle)
                    # print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(),
                    #                                                      "pytorch-" + TASK + "-best"))

            # Write Depth logs
            with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
                epoch_log.write('Train: {}, {:.5f}, {:.5f} \n'.format(epoch, train_loss, train_abs_err))
            mlflow.log_artifact(os.path.join(LOG_FILE, 'log_results.txt'))

        elif TASK == 'depth_segmentation':
            train_loss, train_abs_err, train_rel_err, train_acc, train_miou = static_single_task_trainer(epoch,
                                                                                                         criterion,
                                                                                                         semisup_loss,
                                                                                                         unsup_loss,
                                                                                                         train_dataloader,
                                                                                                         model,
                                                                                                         model_opt,
                                                                                                         scheduler,
                                                                                                         TASK, LOG_FILE)
            metrics['train_loss'].append(train_loss)
            metrics['train_abs_error'].append(train_abs_err)
            metrics['train_rel_error'].append(train_rel_err)
            metrics['train_acc'].append(train_acc)
            metrics['train_miou'].append(train_miou)
            print('Train: epoch {} loss: {:.4f}, abs error: {:.4f}, '
                  'rel error: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                        train_loss,
                                                                        train_abs_err,
                                                                        train_rel_err,
                                                                        train_acc,
                                                                        train_miou))

            # Validate
            if epoch % cfg['training']['val_interval'] == 0:
                print('--- Validation ---')
                val_loss, val_abs_err, val_rel_err, val_acc, val_miou = static_test_single_task(epoch, criterion, semisup_loss, unsup_loss,
                                                                                                val_dataloader, model, TASK, SAMPLES_PATH,
                                                                                                cfg, save_val_imgs=True)
                metrics['val_abs_error'].append(val_abs_err)
                metrics['val_loss'].append(val_loss)
                metrics['val_rel_error'].append(val_rel_err)
                metrics['val_acc'].append(val_acc)
                metrics['val_miou'].append(val_miou)
                print('Validation: epoch {} val loss: {:.4f}, abs error: {:.4f}, '
                      'rel error: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                            val_loss,
                                                                            val_abs_err,
                                                                            val_rel_err,
                                                                            val_acc,
                                                                            val_miou))
                # Save latest validation checkpoint
                path_save_model = os.path.join(MODEL_SAVE_PATH, 'latest_val_checkpoint.pth.tar')
                save_ckpt(path_save_model, model, model_opt, scheduler, metrics, val_miou, epoch)
                # print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(),
                #                                                      "pytorch-" + TASK + "-trained"))

                with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
                    epoch_log.write('Train: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n'.format(epoch, val_loss,
                                                                                          val_abs_err,
                                                                                          val_acc, val_miou))
                # Save best model to file
                if val_abs_err < lowest_depth_error and val_miou > best_miou:
                    print('Val error improved from {:.4f} to {:.4f}.'.format(lowest_depth_error, val_abs_err))
                    lowest_depth_error = val_abs_err
                    best_miou = val_miou
                    path_save_model = os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth.tar')
                    save_ckpt(path_save_model, model, model_opt, scheduler, metrics, val_miou, epoch)
                    mlflow.log_metric('best_abs_error', lowest_depth_error)
                    mlflow.log_metric('best_miou', val_miou)
                    # mlflow.pytorch.log_model(model, artifact_path="pytorch-" + TASK + "-best", pickle_module=pickle)
                    # print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(),
                    #                                                      "pytorch-" + TASK + "-best"))
            # Write segmentation logs
            with open(os.path.join(LOG_FILE, 'log_results.txt'), 'a') as epoch_log:
                epoch_log.write('Train: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n'.format(epoch, train_loss, train_abs_err,
                                                                                      train_acc, train_miou))
            # mlflow.log_artifact(os.path.join(LOG_FILE, 'log_results.txt'))

        # Track the metrics in mlflow
        for key, value in metrics.items():
            # get the most recent metric
            mlflow.log_metric(key, value[-1])

    val_epochs = len(metrics['val_loss'])
    plot_learning_curves(metrics, EPOCHS, val_epochs, SAMPLES_PATH, TASK)
    #plot_metrics_curves(metrics, EPOCHS, val_epochs, SAMPLES_PATH, TASK)
    time_elapsed = time.time() - since

    # Since the model was logged as an artifact, it can be loaded to make predictions
    print("\nLoading model to make predictions on test set")
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth.tar')):
        loaded_model = torch.load(os.path.join(MODEL_SAVE_PATH, 'best_checkpoint.pth.tar'))
    else:
        loaded_model = torch.load(os.path.join(MODEL_SAVE_PATH, 'latest_val_checkpoint.pth.tar'))
    # loaded_model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-" + TASK + "-best"))

    # run on test set
    print('--- Testing ---')
    if TASK == 'segmentation':
        metrics_test = {'test_loss': [],
                        'test_acc': [],
                        'test_miou': []}
        # test set
        test_acc, test_loss, test_miou = static_test_single_task(0, criterion, semisup_loss, unsup_loss,
                                                                 val_dataloader, model, TASK, SAMPLES_PATH,
                                                                 cfg, save_val_imgs=False)
        metrics_test['test_acc'].append(test_acc)
        metrics_test['test_loss'].append(test_loss)
        metrics_test['test_miou'].append(test_miou)
        print('Test: epoch {}, loss: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch, test_loss, test_acc, test_miou))

    elif TASK == 'depth':
        metrics_test = {'test_loss': [],
                        'test_abs_error': [],
                        'test_rel_error': []}
        test_rel_error, test_abs_error, test_loss = static_test_single_task(0, criterion, semisup_loss, unsup_loss,
                                                                            val_dataloader, model, TASK, SAMPLES_PATH,
                                                                            cfg, save_val_imgs=False)
        metrics_test['test_abs_error'].append(test_abs_error)
        metrics_test['test_loss'].append(test_loss)
        metrics_test['test_rel_error'].append(test_rel_error)

        print('Test: epoch {}, loss: {:.4f}, test abs error: {:.4f}, test rel error: {:.4f}'.format(epoch,
                                                                                                  test_loss,
                                                                                                  test_abs_error,
                                                                                                  test_rel_error))
    elif TASK == 'depth_segmentation':
        metrics_test = {'test_loss': [],
                        'test_abs_error': [],
                        'test_rel_error': [],
                        'test_acc': [],
                        'test_miou': []}
        test_loss, test_abs_error, test_rel_error, test_acc, test_miou = static_test_single_task(0, criterion, semisup_loss, unsup_loss,
                                                                                                 val_dataloader, model, TASK, SAMPLES_PATH,
                                                                                                 cfg, save_val_imgs=False)
        metrics_test['test_abs_error'].append(test_abs_error)
        metrics_test['test_loss'].append(test_loss)
        metrics_test['test_acc'].append(test_acc)
        metrics_test['test_miou'].append(test_miou)

        print('Test: epoch {}, loss: {:.4f}, abs error: {:.4f}, '
              'rel error: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                    test_loss,
                                                                    test_abs_error,
                                                                    test_rel_error,
                                                                    test_acc,
                                                                    test_miou))
    # Track the metrics in mlflow
    # for key, value in metrics_test.items():
    #     mlflow.log_metric(key, value[-1])
    # # Extract a few examples from the test dataset to evaluate on
    # eval_data, eval_labels, eval_depth = next(iter(test_dataloader))
    # # Make a few predictions
    # predictions = loaded_model(eval_data).data.max(1)[1]
    # template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
    # print("\nSample predictions")
    # for index in range(5):
    #     print(template.format(index, eval_labels[index], predictions[index]))

print('Done!!')