import numpy as np
import torch
import argparse
# import itertools
import tempfile
import oyaml as yaml

# from tensorboardX import SummaryWriter
# from torchvision.datasets import Cityscapes
from loader.cityscapes_loader import cityscapesLoader
from loader.city_loader import cityscapesLoader2
# from loader.video_dataset import *
from utils.sort_dataset import *
# from loader.nyuv2_dataloader import NYUV2
from models.decoder import Decoder, decoder
from models.mtl_model import MultiTaskModel
from models.static_model import StaticTaskModel
from models.deeplabv3_encoder import DeepLabv3
from utils.train_helpers import *
from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

# commandline arguments
parser = argparse.ArgumentParser(description='Multi-Task Temporal w/ Partial annotation')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use, default 2')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum, default False')
parser.add_argument('--model_save_path', type=str, default='.', help='Path to the saved models')
# parser.add_argument('--sgdadjust', default=1, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
# parser.add_argument('--window_size', type=int, default=5, help='ILA module computes on a window size')
parser.add_argument("--config", default='configs/ucl_cluster/cityscapes_config_static',
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
# Slow and fast encoder backbone
BACKBONE = cfg["model"]["backbone"]["encoder"]
# multi_optim = args.multi
# use_nesterov = args.nesterov
DATASET_PATH = cfg["data"]["path"]
CLASS_TASKS = cfg["model"]["task_classes"]
frames_per_segment = cfg["model"]["frames_per_segment"]

TASK = cfg["model"]["tasks"]
path_n = cfg["model"]["path_num"]
# data_loader = get_loader(cfg["data"]["dataset"])
data_path = cfg["data"]["path"]
image_size = 256
train_augmentations = torch.nn.Sequential(transforms.Resize(size=(256, 512)),
                                          # transforms.RandomCrop(size=(256, 512)),
                                          transforms.RandomHorizontalFlip(p=0.5)
                                          # transforms.Normalize(mean=(123.675, 116.28, 103.53),
                                          #                      std=(58.395, 57.12, 57.375)),
                                          # transforms.Pad(padding=(256, 512))
                                          )

val_augmentations = torch.nn.Sequential(transforms.Resize(size=(256, 512)),
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
                            test_mode=False,
                            model_name=None,
                            path_num=path_n)
train_dataloader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=cfg["training"]["batch_size"],
                                               shuffle=False,
                                               num_workers=cfg["training"]["n_workers"])
test_dataloader = torch.utils.data.DataLoader(dataset=val_set,
                                              batch_size=cfg["validating"]["batch_size"],
                                              shuffle=False,
                                              num_workers=cfg["validating"]["n_workers"])
print('Initialising Model')
# Initializing encoder
deeplabv3_backbone = cfg["model"]["backbone"]["encoder"]["resnet_version"]
enc = DeepLabv3(deeplabv3_backbone).to(device)
# enc_optimizer = optim.SGD(enc.parameters(),
#                           lr=cfg["training"]["optimizer"]["lr0"],
#                           momentum=cfg["training"]["optimizer"]["momentum"],
#                           weight_decay=cfg["training"]["optimizer"]["wd"])

# initialise decoders (one for each task)
dec = decoder(image_size, CLASS_TASKS, SynchronizedBatchNorm1d).to(device)
# initialise multi (or single) - task model
model = StaticTaskModel(enc, dec).to(device)
model_opt = optim.Adam(model.parameters(), lr=cfg["training"]["optimizer"]["lr0"])
scheduler = optim.lr_scheduler.StepLR(model_opt,
                                      step_size=cfg["training"]["scheduler"]["step"],
                                      gamma=cfg["training"]["scheduler"]["gamma"])
# model = model.to(device)
model = nn.DataParallel(model)

# directory name to save the models
MODEL_SAVE_PATH = os.path.join(args.model_save_path, 'Results', 'Model', 'slowfast')
LOG_FILE = os.path.join(args.model_save_path, 'Results', 'Model', 'logs')
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
SAMPLES_PATH = os.path.join(args.model_save_path, 'Results', 'Samples', 'slowfast')
if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)

# LOG_FILE_NAME = 'log.txt'
# logging.basicConfig(filename=os.path.join(LOG_FILE, LOG_FILE_NAME),
#                         filemode='w',
#                         format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                         datefmt='%H:%M:%S',
#                         level=logging.INFO)

# Load the pretrained model
# The backbones are pre-trained on ImageNet
# if PRE_TRAIN is not None:
#     pretrained_dict = torch.load(PRE_TRAIN, map_location='cpu')
#     try:
#         model_dict = encoder.module.state_dict()
#     except AttributeError:
#         model_dict = encoder.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     print("load pretrain model")
#     model_dict.update(pretrained_dict)
#     encoder.load_state_dict(model_dict)

# print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')
train_batch = len(train_dataloader)
test_batch = len(test_dataloader)

# criterion = F.nll_loss()
# criterion = DiceLoss()
print('Beginning training...')
with mlflow.start_run():
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    # Create a SummaryWriter to write TensorBoard events locally
    # output_dir = dirpath = tempfile.mkdtemp()
    output_file = tempfile.mkdtemp()
    writer = SummaryWriter(output_file)
    print("Writing TensorBoard events locally to %s\n" % output_file)
    # perform the training
    avg_cost_epoch = np.zeros([EPOCHS, 24], dtype=np.float32)
    # lambda_weight = np.ones([2, EPOCHS])
    lowest_error_decoders = float('inf')
    for epoch in range(EPOCHS):
        train_loss, cost_train, avg_cost_train = static_single_task_trainer(epoch, train_dataloader, model,
                                                                            model_opt, TASK, writer)

        test_loss, cost_test, avg_cost_test = static_test_single_task(epoch, test_dataloader, model, writer,
                                                                      scheduler, TASK, SAMPLES_PATH)

        avg_cost_epoch[epoch, :12] = avg_cost_train
        avg_cost_epoch[epoch, 12:] = avg_cost_test
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} || '
               .format(epoch, avg_cost_epoch[epoch, 0], avg_cost_epoch[epoch, 1], avg_cost_epoch[epoch, 2],
                       avg_cost_epoch[epoch, 3], avg_cost_epoch[epoch, 4], avg_cost_epoch[epoch, 5],
                       avg_cost_epoch[epoch, 6], avg_cost_epoch[epoch, 7], avg_cost_epoch[epoch, 8],
                       avg_cost_epoch[epoch, 9], avg_cost_epoch[epoch, 10], avg_cost_epoch[epoch, 11],
                       avg_cost_epoch[epoch, 12], avg_cost_epoch[epoch, 13], avg_cost_epoch[epoch, 14],
                       avg_cost_epoch[epoch, 15], avg_cost_epoch[epoch, 16], avg_cost_epoch[epoch, 17],
                       avg_cost_epoch[epoch, 18], avg_cost_epoch[epoch, 19], avg_cost_epoch[epoch, 20],
                       avg_cost_epoch[epoch, 21], avg_cost_epoch[epoch, 22], avg_cost_epoch[epoch, 23]))

        # save best decoders
        if avg_cost_test < lowest_error_decoders:
            # save the best decoder
            lowest_error_decoders = avg_cost_test
            torch.save(encoder_slow.state_dict(), os.path.join(MODEL_SAVE_PATH, args.model +
                                                               '_enc_slow' + '_best_accuracy.pt'))
            torch.save(encoder_fast.state_dict(), os.path.join(MODEL_SAVE_PATH, args.model +
                                                               '_enc_slow' + '_best_accuracy.pt'))
            # torch.save(dec_seg.state_dict(), os.path.join(MODEL_SAVE_PATH, args.model +
            #                                               '_dec_' + str(0) + '_best_accuracy.pt'))
            torch.save(dec.state_dict(),
                       os.path.join(MODEL_SAVE_PATH, args.model + '_dec_' + str(decoder) + '_best_accuracy.pt'))

    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(output_dir, artifact_path="events")
    print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" % os.path.join(mlflow.get_artifact_uri(), "events"))

    # Log the model as an artifact of the MLflow run.
    print("\nLogging the trained model as a run artifact...")
    mlflow.pytorch.log_model(model, artifact_path="pytorch-model", pickle_module=pickle)
    print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model"))

    # Since the model was logged as an artifact, it can be loaded to make predictions
    loaded_model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-model"))

    # Extract a few examples from the test dataset to evaluate on
    eval_data, eval_labels, eval_depth = next(iter(test_dataloader))
    # Make a few predictions
    predictions = loaded_model(eval_data).data.max(1)[1]
    template = 'Sample {} : Ground truth is "{}", model prediction is "{}"'
    print("\nSample predictions")
    for index in range(5):
        print(template.format(index, eval_labels[index], predictions[index]))

print('Done!!')