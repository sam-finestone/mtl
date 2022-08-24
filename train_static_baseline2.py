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
from loader.city_loader import cityscapesLoader2
# from loader.video_dataset import *
from utils.sort_dataset import *
# from loader.nyuv2_dataloader import NYUV2
from models.decoder import Decoder, SegDecoder, DepthDecoder, MultiDecoder
from models.mtl_model import MultiTaskModel
from models.static_model import StaticTaskModel
from models.deeplabv3_encoder import DeepLabv3
from utils.train_helpers import *
from utils.static_helpers import static_single_task_trainer, static_test_single_task
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback
from utils.metrics import plot_learning_curves
from loss.loss import InverseDepthL1Loss
from scheduler import get_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

# commandline arguments
parser = argparse.ArgumentParser(description='Multi-Task Temporal w/ Partial annotation')
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
# uncomment for segmentation run
# parser.add_argument("--config", default='configs/medtronic_cluster/static_cityscape_config_seg',
#                     nargs="?", type=str, help="Configuration file to use")

# uncomment for depth run
parser.add_argument("--config", default='configs/medtronic_cluster/static_cityscape_config_depth',
                    nargs="?", type=str, help="Configuration file to use")

# uncomment for both tasks
# parser.add_argument("--config", default='configs/medtronic_cluster/static_cityscape_config_both',
#                     nargs="?", type=str, help="Configuration file to use")


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
# Slow and fast encoder backbone
BACKBONE = cfg["model"]["backbone"]["encoder"]
# multi_optim = args.multi
# use_nesterov = args.nesterov
DATASET_PATH = cfg["data"]["path"]
CLASS_TASKS = cfg["model"]["task_classes"]
frames_per_segment = cfg["model"]["frames_per_segment"]
path_n = cfg["model"]["path_num"]
data_path = cfg["data"]["path"]
input_dim_decoder = 256
TASK = cfg["model"]["tasks"]

# Initialize mlflow
NAME_EXPERIMENT = 'experiment_static_' + TASK
# mlflow.set_tracking_uri("http://10.167.61.230:5000")
# mlflow.set_tracking_uri("https://mlflow-dev.touchsurgery.com")
mlflow.set_experiment(experiment_name=NAME_EXPERIMENT)

train_augmentations = torch.nn.Sequential(transforms.Resize(size=(128, 256)),
                                          # transforms.RandomCrop(size=(256, 512)),
                                          transforms.RandomHorizontalFlip(p=0.5)
                                          # transforms.Normalize(mean=(123.675, 116.28, 103.53),
                                          #                      std=(58.395, 57.12, 57.375)),
                                          # transforms.Pad(padding=(256, 512))
                                          )

val_augmentations = torch.nn.Sequential(transforms.Resize(size=(128, 256)),
                                        # transforms.RandomCrop(size=(256, 512)),
                                        transforms.RandomHorizontalFlip(p=0.5)
                                        # transforms.Normalize(mean=(123.675, 116.28, 103.53),
                                        #                      std=(58.395, 57.12, 57.375)),
                                        # transforms.Pad(padding=(256, 512))
                                        )

# Load the dataloaders
print('Preprocessing ' + cfg["data"]["dataset"])
train_set = cityscapesLoader2(DATASET_PATH,
                              split=cfg["data"]["train_split"],
                              augmentations=train_augmentations,
                              test_mode=False,
                              model_name=None,
                              path_num=path_n)
val_set = cityscapesLoader2(DATASET_PATH,
                            split=cfg["data"]["val_split"],
                            augmentations=val_augmentations,
                            test_mode=True,
                            model_name=None,
                            path_num=path_n)
test_set = cityscapesLoader2(DATASET_PATH,
                            split=cfg["data"]["test_split"],
                            augmentations=val_augmentations,
                            test_mode=True,
                            model_name=None,
                            path_num=path_n)

classLabels = val_set.class_names
# print(len(classLabels))
validClasses = val_set.valid_classes
mask_colors = val_set.colors
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
print('Initialising Model')
# Initializing encoder
deeplabv3_backbone = cfg["model"]["backbone"]["encoder"]["resnet_version"]
enc = DeepLabv3(deeplabv3_backbone).to(device)
# enc_optimizer = optim.SGD(enc.parameters(),
#                           lr=cfg["training"]["optimizer"]["lr0"],
#                           momentum=cfg["training"]["optimizer"]["momentum"],
#                           weight_decay=cfg["training"]["optimizer"]["wd"])

# initialise decoders (one for each task)
drop_out = cfg["model"]["dropout"]
if TASK == 'segmentation':
    dec = SegDecoder(input_dim_decoder, CLASS_TASKS, drop_out, SynchronizedBatchNorm1d).to(device)
elif TASK == 'depth':
    dec = DepthDecoder(input_dim_decoder, CLASS_TASKS, drop_out, SynchronizedBatchNorm1d).to(device)
elif TASK == 'depth_segmentation':
    dec = MultiDecoder(input_dim_decoder, CLASS_TASKS, drop_out, SynchronizedBatchNorm1d).to(device)

# initialise multi (or single) - task model
model = StaticTaskModel(enc, dec, TASK).to(device)

# Push model to GPU
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).to(device)
    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))


model_opt = optim.Adam(model.parameters(), lr=cfg["training"]["optimizer"]["lr"])
# scheduler = get_scheduler(model_opt, cfg['training']['lr_schedule'])
scheduler = optim.lr_scheduler.StepLR(model_opt,
                                      step_size=cfg["training"]["scheduler"]["step"],
                                      gamma=cfg["training"]["scheduler"]["gamma"])

# directory name to save the models
MODEL_SAVE_PATH = os.path.join(args.model_save_path, 'Model', 'Checkpoints', TASK)
LOG_FILE = os.path.join(args.model_save_path, 'Logs', TASK)
SAMPLES_PATH = os.path.join(args.model_save_path, 'Model', 'Sample', TASK)

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
    with open(os.path.join(LOG_FILE, 'log_epoch.txt'), 'a') as epoch_log:
        epoch_log.write('epoch, train loss, val loss, train acc, val acc, train miou, val miou\n')

    with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as f:
        f.write('Epoch, Batch Index, train loss, train acc, train miou\n')

if TASK == 'depth':
    # for each epoch record important metrics
    with open(os.path.join(LOG_FILE, 'log_epoch.txt'), 'a') as epoch_log:
        epoch_log.write('epoch, train loss, val loss, train error, val error\n')

    with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as f:
        f.write('Epoch, Batch Index, train loss, abs_err, rel_err\n')

if TASK == 'depth_segmentation':
    # for each epoch record important metrics
    with open(os.path.join(LOG_FILE, 'log_epoch.txt'), 'a') as epoch_log:
        epoch_log.write('epoch, train loss, train acc, train abs error, train rel error, '
                        'val loss, val acc, val miou, val abs error, val rel error \n')

    with open(os.path.join(LOG_FILE, 'log_train_batch.txt'), 'a') as f:
        f.write('Epoch, Batch Index, train loss, train acc, train abs error, train rel error \n')

LOG_FILE_NAME = 'log.txt'

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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_set.ignore_index)

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
    criterion = InverseDepthL1Loss()

elif TASK == 'depth_segmentation':
    if not os.path.exists(os.path.join(SAMPLES_PATH, 'images')):
        os.makedirs(os.path.join(SAMPLES_PATH, 'images'))
    # Initialize metrics
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
    criterion = [InverseDepthL1Loss(), torch.nn.CrossEntropyLoss(ignore_index=train_set.ignore_index)]


print('Beginning training...')
with mlflow.start_run():

    # Logging the important parameters for the model
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('task', TASK)
    mlflow.log_param('training_batch_size', cfg["training"]["batch_size"])
    mlflow.log_param('number_of_workers', cfg["training"]["n_workers"])
    mlflow.log_param('backbone', BACKBONE)
    mlflow.log_param('enocoder_backbone_resnet', deeplabv3_backbone)
    mlflow.log_param('encoder_pretrained', cfg["training"]["pretrain"])
    mlflow.log_param('dropout', drop_out)
    mlflow.log_param('model_optimizer', cfg["training"]["optimizer"]["name"])
    mlflow.log_param('model_optimizer_lr', cfg["training"]["optimizer"]["lr"])
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    # # Create a SummaryWriter to write TensorBoard events locally
    # # output_dir = dirpath = tempfile.mkdtemp()
    # output_file = tempfile.mkdtemp()
    # writer = SummaryWriter(output_file)
    # print("Writing TensorBoard events locally to %s\n" % output_file)
    # lambda_weight = np.ones([2, EPOCHS])

    lowest_depth_error = float('inf')
    for epoch in range(EPOCHS):

        print('--- Training ---')
        if TASK == 'segmentation':
            train_loss, train_acc, train_miou = static_single_task_trainer(epoch, criterion, train_dataloader, model,
                                                                           model_opt, scheduler, TASK, LOG_FILE)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['train_miou'].append(train_miou)
            print('Epoch {} train loss: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                                  train_loss,
                                                                                  train_acc,
                                                                                  train_miou))

            # Validate
            print('--- Validation ---')
            val_acc, val_loss, val_miou = static_test_single_task(epoch, criterion, val_dataloader,
                                                                  model, TASK, classLabels,
                                                                  validClasses, SAMPLES_PATH, void=0,
                                                                  maskColors=mask_colors, args=args)
            metrics['val_acc'].append(val_acc)
            metrics['val_loss'].append(val_loss)
            metrics['val_miou'].append(val_miou)

            print('Epoch {} val loss: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch, val_loss, val_acc, val_miou))
            # Write segmentation logs
            with open(os.path.join(LOG_FILE, 'log_epoch.txt'), 'a') as epoch_log:
                epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                                epoch, train_loss, val_loss, train_acc, val_acc, val_miou))
            mlflow.log_artifact(os.path.join(LOG_FILE, 'log_epoch.txt'))

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_opt.state_dict(),
                'best_miou': best_miou,
                'metrics': metrics,
            }, MODEL_SAVE_PATH + '/checkpoint.pth.tar')

            # Save best model to file
            if val_miou > best_miou:
                print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, val_miou))
                best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, MODEL_SAVE_PATH + '/best_weights.pth.tar')
                mlflow.log_metric('best_miou', best_miou)
                print("\nLogging the trained model as a run artifact...")
                mlflow.pytorch.log_model(model, artifact_path="pytorch-"+TASK+"-best", pickle_module=pickle)

        elif TASK == 'depth':
            train_loss, train_abs_err, train_rel_err = static_single_task_trainer(epoch, criterion, train_dataloader,
                                                                                  model, model_opt, scheduler, TASK,
                                                                                  LOG_FILE)
            metrics['train_loss'].append(train_loss)
            metrics['train_abs_error'].append(train_abs_err)
            metrics['train_rel_error'].append(train_rel_err)

            print('Epoch {} train loss: {:.4f}, abs error: {:.4f}, val rel error: {:.4f}'.format(epoch,
                                                                                                 train_loss,
                                                                                                 train_abs_err,
                                                                                                 train_rel_err))

            # Validate
            print('--- Validation ---')
            val_rel_error, val_abs_error, val_loss = static_test_single_task(epoch, criterion, val_dataloader, model,
                                                                             TASK, classLabels, validClasses,
                                                                             SAMPLES_PATH, void=0, maskColors=None,
                                                                             args=None)
            metrics['val_abs_error'].append(val_abs_error)
            metrics['val_loss'].append(val_loss)
            metrics['val_rel_error'].append(val_rel_error)

            print('Epoch {} val loss: {:.4f}, val abs error: {:.4f}, val rel error: {:.4f}'.format(epoch,
                                                                                                   val_loss,
                                                                                                   val_abs_error,
                                                                                                   val_rel_error))
            # Write segmentation logs
            with open(os.path.join(LOG_FILE, 'log_epoch.txt'), 'a') as epoch_log:
                epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f} \n'.format(
                    epoch, train_loss, val_loss, train_abs_err, val_abs_error, train_rel_err, val_rel_error))
            mlflow.log_artifact(os.path.join(LOG_FILE, 'log_epoch.txt'))
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_opt.state_dict(),
                'val_abs_error': val_abs_error,
                'metrics': metrics,
            }, MODEL_SAVE_PATH + '/checkpoint.pth.tar')

            # # Since the model was logged as an artifact, it can be loaded to make predictions
            print("\nLogging the trained model as a run artifact...")
            mlflow.pytorch.log_model(model, artifact_path="pytorch-"+TASK+"-trained", pickle_module=pickle)
            print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-" + TASK + "-trained"))

            # Save best model to file
            if val_abs_error < lowest_depth_error:
                print('Val error improved from {:.4f} to {:.4f}.'.format(lowest_depth_error, val_abs_error))
                lowest_depth_error = val_abs_error
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, MODEL_SAVE_PATH + '/best_weights.pth.tar')
                mlflow.log_metric('lowest_depth_error', lowest_depth_error)
                print("\nLogging the trained model as a run artifact...")
                mlflow.pytorch.log_model(model, artifact_path="pytorch-"+TASK+"-best", pickle_module=pickle)

        elif TASK == 'depth_segmentation':
            train_loss, train_abs_err, train_rel_err, train_acc, train_miou = static_single_task_trainer(epoch,
                                                                                                         criterion,
                                                                                                         train_dataloader,
                                                                                                         model,
                                                                                                         model_opt,
                                                                                                         scheduler,
                                                                                                         TASK,
                                                                                                         LOG_FILE)
            metrics['train_loss'].append(train_loss)
            metrics['train_abs_error'].append(train_abs_err)
            metrics['train_rel_error'].append(train_rel_err)
            metrics['train_acc'].append(train_acc)
            metrics['train_miou'].append(train_miou)
            print('Epoch {} train loss: {:.4f}, abs error: {:.4f}, '
                  'rel error: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                        train_loss,
                                                                        train_abs_err,
                                                                        train_rel_err,
                                                                        train_acc,
                                                                        train_miou))

            # Validate
            print('--- Validation ---')
            val_loss, val_abs_err, val_rel_err, val_acc, val_miou = static_test_single_task(epoch, criterion,
                                                                                            val_dataloader, model,
                                                                                            TASK, classLabels,
                                                                                            validClasses,
                                                                                            SAMPLES_PATH,
                                                                                            void=0, maskColors=None,
                                                                                            args=None)
            metrics['val_abs_error'].append(val_abs_err)
            metrics['val_loss'].append(val_loss)
            metrics['val_rel_error'].append(val_rel_err)
            metrics['val_acc'].append(val_acc)
            metrics['val_miou'].append(val_miou)
            print('Epoch {} val loss: {:.4f}, abs error: {:.4f}, '
                  'rel error: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                        val_loss,
                                                                        val_abs_err,
                                                                        val_rel_err,
                                                                        val_acc,
                                                                        val_miou))
            # Write segmentation logs
            with open(os.path.join(LOG_FILE, 'log_epoch.txt'), 'a') as epoch_log:
                epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, '
                                '{:.5f}, {:.5f} \n'.format(epoch, train_loss, val_loss, train_abs_err,
                                                           val_abs_err, train_rel_err, val_rel_err,
                                                           train_acc, val_acc, train_miou, val_miou))
            mlflow.log_artifact(os.path.join(LOG_FILE, 'log_epoch.txt'))
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_opt.state_dict(),
                'val_abs_error': val_abs_error,
                'metrics': metrics,
            }, MODEL_SAVE_PATH + '/checkpoint.pth.tar')

            # # Since the model was logged as an artifact, it can be loaded to make predictions
            print("\nLogging the trained model as a run artifact...")
            mlflow.pytorch.log_model(model, artifact_path="pytorch-" + TASK + "-trained", pickle_module=pickle)
            print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-" + TASK + "-trained"))

            # Save best model to file
            if val_abs_error < lowest_depth_error:
                print('Val error improved from {:.4f} to {:.4f}.'.format(lowest_depth_error, val_abs_error))
                lowest_depth_error = val_abs_error
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, MODEL_SAVE_PATH + '/best_weights.pth.tar')
                mlflow.log_metric('lowest_depth_error', lowest_depth_error)
                print("\nLogging the trained model as a run artifact...")
                mlflow.pytorch.log_model(model, artifact_path="pytorch-" + TASK + "-best", pickle_module=pickle)

        # Track the metrics in mlflow
        for key, value in metrics.items():
            # get the most recent metric
            mlflow.log_metric(key, value[-1])

    plot_learning_curves(metrics, EPOCHS, SAMPLES_PATH, TASK)
    time_elapsed = time.time() - since

    # Since the model was logged as an artifact, it can be loaded to make predictions
    print("\nLoading model to make predictions on test set")
    loaded_model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-"+TASK+"-best"))

    # run on test set
    print('--- Testing ---')
    if TASK == 'segmentation':
        metrics_test = {'test_loss': [],
                        'test_acc': [],
                        'test_miou': []}
        # test set
        test_acc, test_loss, test_miou = static_test_single_task(0, criterion, test_dataloader,
                                                                 loaded_model, TASK, classLabels,
                                                                 validClasses, SAMPLES_PATH, void=0,
                                                                 maskColors=None, args=None)
        metrics_test['test_acc'].append(test_acc)
        metrics_test['test_loss'].append(test_loss)
        metrics_test['test_miou'].append(test_miou)
        print('Epoch {} test loss: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch, test_loss, test_acc, test_miou))

    elif TASK == 'depth':
        metrics_test = {'test_loss': [],
                        'test_abs_error': [],
                        'test_rel_error': []}
        test_rel_error, test_abs_error, test_loss = static_test_single_task(0, criterion, test_dataloader,
                                                                            loaded_model, TASK, classLabels,
                                                                            validClasses, SAMPLES_PATH, void=0,
                                                                            maskColors=None, args=None)
        metrics_test['test_abs_error'].append(test_abs_error)
        metrics_test['test_loss'].append(test_loss)
        metrics_test['test_rel_error'].append(test_rel_error)

        print('Epoch {} Test loss: {:.4f}, test abs error: {:.4f}, test rel error: {:.4f}'.format(epoch,
                                                                                                  test_loss,
                                                                                                  test_abs_error,
                                                                                                  test_rel_error))
    elif TASK == 'depth_segmentation':
        metrics_test = {'test_loss': [],
                        'test_abs_error': [],
                        'test_rel_error': [],
                        'test_acc': [],
                        'test_miou': []}
        test_loss, test_abs_error, test_rel_error, test_acc, test_miou = static_test_single_task(0, criterion,
                                                                                                 test_dataloader,
                                                                                                 loaded_model, TASK,
                                                                                                 classLabels,
                                                                                                 validClasses,
                                                                                                 SAMPLES_PATH, void=0,
                                                                                                 maskColors=None,
                                                                                                 args=None)
        metrics_test['test_abs_error'].append(test_abs_error)
        metrics_test['test_loss'].append(test_loss)
        metrics_test['test_acc'].append(test_acc)
        metrics_test['test_miou'].append(test_miou)

        print('Epoch {} Test loss: {:.4f}, abs error: {:.4f}, '
              'rel error: {:.4f}, acc: {:.4f}, miou: {:.4f}'.format(epoch,
                                                                    test_loss,
                                                                    test_abs_error,
                                                                    test_rel_error,
                                                                    test_acc,
                                                                    test_miou))
    # Track the metrics in mlflow
    for key, value in metrics_test.items():
        mlflow.log_metric(key, value[-1])
    # # Extract a few examples from the test dataset to evaluate on
    # eval_data, eval_labels, eval_depth = next(iter(test_dataloader))
    # # Make a few predictions
    # predictions = loaded_model(eval_data).data.max(1)[1]
    # template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
    # print("\nSample predictions")
    # for index in range(5):
    #     print(template.format(index, eval_labels[index], predictions[index]))

print('Done!!')