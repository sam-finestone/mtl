import numpy as np
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import os

# mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu), acc


def depth_error(x_pred, x_output):
    device = x_pred.device
    # invalid_idx = -1
    # binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    binary_mask = (torch.sum(x_output, dim=1, keepdim=True) != -1).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()

def depth_error2(pred, gt):
    invalid_idx = -1
    # valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
    valid_mask = (gt != invalid_idx).to(pred.device)
    abs_err = torch.mean(torch.abs(pred - gt).masked_select(valid_mask)).item()
    # rel_err = torch.mean((torch.abs(pred - gt)/gt).masked_select(valid_mask)).item()
    # rel_err = torch.mean((torch.abs(pred - gt)/.masked_select(valid_mask)).item()
    # rel_err = torch.mean(torch.abs(pred - gt).masked_select(valid_mask)).item()
    return abs_err, 0


"""
====================
Classes for logging and progress printing.
====================
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


"""
====================
iouCalc: Calculates IoU scores of dataset per individual batch.
Arguments:
    - labelNames: list of class names
    - validClasses: list of valid class ids
    - voidClass: class id of void class (class ignored for calculating metrics)
====================
"""


class iouCalc():

    def __init__(self, classLabels, validClasses, voidClass=None):
        assert len(classLabels) == len(validClasses), 'Number of class ids and names must be equal'
        self.classLabels = classLabels
        self.validClasses = validClasses
        self.voidClass = voidClass
        self.evalClasses = [l for l in validClasses if l != voidClass]
        self.class_map = dict(zip(self.validClasses, range(19)))
        self.perImageStats = []
        self.nbPixels = 0
        self.confMatrix = np.zeros(shape=(len(self.validClasses)+1, len(self.validClasses)+1), dtype=np.ulonglong)

        # Init IoU log files
        self.headerStr = 'epoch, '
        for label in self.classLabels:
            if label.lower() != 'void':
                self.headerStr += label + ', '

    def clear(self):
        self.perImageStats = []
        self.nbPixels = 0
        self.confMatrix = np.zeros(shape=(len(self.validClasses)+1, len(self.validClasses)+1), dtype=np.ulonglong)

    def getIouScoreForLabel(self, label):
        # Calculate and return IOU score for a particular label (train_id)
        if label == self.voidClass:
            return float('nan')

        # the number of true positive pixels for this label
        # the entry on the diagonal of the confusion matrix
        tp = np.longlong(self.confMatrix[label, label])

        # the number of false negative pixels for this label
        # the row sum of the matching row in the confusion matrix
        # minus the diagonal entry
        fn = np.longlong(self.confMatrix[label, :].sum()) - tp

        # the number of false positive pixels for this labels
        # Only pixels that are not on a pixel with ground truth label that is ignored
        # The column sum of the corresponding column in the confusion matrix
        # without the ignored rows and without the actual label of interest
        notIgnored = [l for l in self.validClasses if not l == self.voidClass and not l == label]
        fp = np.longlong(self.confMatrix[notIgnored, label].sum())

        # the denominator of the IOU score
        denom = (tp + fp + fn)
        if denom == 0:
            return float('nan')

        # return IOU
        return float(tp) / denom

    def evaluateBatch(self, predictionBatch, groundTruthBatch):
        # Calculate IoU scores for single batch
        predictionBatch = torch.from_numpy(predictionBatch)
        groundTruthBatch = torch.from_numpy(groundTruthBatch)
        assert predictionBatch.size(0) == groundTruthBatch.size(0), \
            'Number of predictions and labels in batch disagree.'

        # Load batch to CPU and convert to numpy arrays
        predictionBatch = predictionBatch.cpu().numpy()
        groundTruthBatch = groundTruthBatch.cpu().numpy()
        for i in range(predictionBatch.shape[0]):
            predictionImg = predictionBatch[i, :, :]
            groundTruthImg = groundTruthBatch[i, :, :]

            assert predictionImg.shape == groundTruthImg.shape, 'Image shapes do not match.'
            assert len(predictionImg.shape) == 2, 'Predicted image has multiple channels.'

            imgWidth = predictionImg.shape[0]
            imgHeight = predictionImg.shape[1]
            nbPixels = imgWidth * imgHeight

            # Evaluate images
            encoding_value = max(groundTruthImg.max(), predictionImg.max()).astype(np.int32) + 1
            encoded = (groundTruthImg.astype(np.int32) * encoding_value) + predictionImg

            values, cnt = np.unique(encoded, return_counts=True)
            for value, c in zip(values, cnt):
                pred_id = value % encoding_value
                gt_id = int((value - pred_id) / encoding_value)
                if not gt_id in self.validClasses:
                    printError('Unknown label with id {:}'.format(gt_id))
                self.confMatrix[gt_id][pred_id] += c

            # Calculate pixel accuracy
            notIgnoredPixels = np.in1d(groundTruthImg, self.evalClasses, invert=True).reshape(groundTruthImg.shape)
            erroneousPixels = np.logical_and(notIgnoredPixels, (predictionImg != groundTruthImg))
            nbNotIgnoredPixels = np.count_nonzero(notIgnoredPixels)
            nbErroneousPixels = np.count_nonzero(erroneousPixels)
            self.perImageStats.append([nbNotIgnoredPixels, nbErroneousPixels])

            self.nbPixels += nbPixels

        return

    def outputScores(self, save_file_path):
        # Output scores over dataset
        assert self.confMatrix.sum() == self.nbPixels, 'Number of analyzed pixels and entries in confusion matrix disagree: confMatrix {}, pixels {}'.format(
            self.confMatrix.sum(), self.nbPixels)

        # Calculate IOU scores on class level from matrix
        classScoreList = []

        # Print class IOU scores
        outStr = 'classes           IoU\n'
        outStr += '---------------------\n'
        for c in self.evalClasses:
            iouScore = self.getIouScoreForLabel(c)
            classScoreList.append(iouScore)
            outStr += '{:<14}: {:>5.3f}\n'.format(self.classLabels[c], iouScore)
        miou = getScoreAverage(classScoreList)
        outStr += '---------------------\n'
        outStr += 'Mean IoU      : {avg:5.3f}\n'.format(avg=miou)
        outStr += '---------------------'

        print(outStr)
        with open(os.path.join(save_file_path, 'IoU_scores.txt'), 'a') as iou_epoch:
            iou_epoch.write(outStr)
        return miou


# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)


def getScoreAverage(scoreList):
    validScores = 0
    scoreSum = 0.0
    for score in scoreList:
        if not np.isnan(score):
            validScores += 1
            scoreSum += score
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores


"""
================
Visualize images
================
"""


def visim(img, args):
    img = img.cpu()
    # Convert image data to visual representation
    img *= torch.tensor(args.dataset_std)[:, None, None]
    img += torch.tensor(args.dataset_mean)[:, None, None]
    npimg = (img.numpy() * 255).astype('uint8')
    if len(npimg.shape) == 3 and npimg.shape[0] == 3:
        npimg = np.transpose(npimg, (1, 2, 0))
    else:
        npimg = npimg[0, :, :]
    return npimg


def vislbl(label, mask_colors):
    label = label.cpu()
    # Convert label data to visual representation
    label = np.array(label.numpy())
    if label.shape[-1] == 1:
        label = label[:, :, 0]

    # Convert train_ids to colors
    # label = torch.apply()
    # print(label[i, j])
    # print(mask_colors)
    converted_to_color = np.zeros((128, 256, 3), dtype=np.uint8)
    for i in range(128):
        for j in range(256):
            converted_to_color[i, j, :] = mask_colors[label[i, j]]
    # label = mask_colors[label]
    return converted_to_color


"""
====================
Plot learning curves
====================
"""


def plot_learning_curves(metrics, epochs, val_epochs, save_img_path, task, ):
    x_train = np.arange(epochs)
    x_val = np.arange(val_epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    if task == 'segmentation':
        ln1 = ax1.plot(x_train, metrics['train_loss'], color='tab:red')
        ln2 = ax1.plot(x_val, metrics['val_loss'], color='tab:red', linestyle='dashed')
        ax1.grid()
        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy')
        ln3 = ax2.plot(x_train, metrics['train_acc'], color='tab:blue')
        ln4 = ax2.plot(x_val, metrics['val_acc'], color='tab:blue', linestyle='dashed')
        ln5 = ax2.plot(x_val, metrics['val_miou'], color='tab:green')
        lns = ln1 + ln2 + ln3 + ln4 + ln5
        plt.legend(lns, ['Train loss', 'Validation loss', 'Train accuracy', 'Validation accuracy', 'Val mIoU'])
        plt.tight_layout()
        plt.savefig(save_img_path + '/learning_curve.png', bbox_inches='tight')

    elif task == 'depth':
        ln1 = ax1.plot(x_train, metrics['train_loss'], color='tab:red')
        ln2 = ax1.plot(x_val, metrics['val_loss'], color='tab:red', linestyle='dashed')
        ax1.grid()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Abs. Error')
        ln3 = ax2.plot(x_train, metrics['train_abs_error'], color='tab:blue')
        ln4 = ax2.plot(x_val, metrics['val_abs_error'], color='tab:blue', linestyle='dashed')
        lns = ln1 + ln2 + ln3 + ln4
        plt.legend(lns, ['Train loss', 'Validation loss', 'Train Absolute Error', 'Validation Absolute Error'])
        plt.tight_layout()
        plt.savefig(save_img_path + '/learning_curve.png', bbox_inches='tight')

    elif task == 'depth_segmentation':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(x_train, metrics['train_loss'], color='tab:blue')
        ax1.plot(x_val, metrics['val_loss'], color='tab:red')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid()
        # ax2 = ax1.twinx()

        ax2.plot(x_train, metrics['train_abs_error'], color='tab:blue')
        ax2.plot(x_val, metrics['val_abs_error'], color='tab:red')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Error')
        ax2.grid()

        ax3.plot(x_train, metrics['train_acc'], color='tab:blue')
        ax3.plot(x_val, metrics['val_acc'], color='tab:blue', linestyle='dashed')
        ax3.plot(x_train, metrics['train_miou'], color='tab:red')
        ax3.plot(x_val, metrics['val_miou'], color='tab:red', linestyle='dashed')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy/mIou')
        ax3.grid()
        # lns = ln1 + ln2 + ln3 + ln4
        # plt.legend(lns, ['Train loss', 'Validation loss', 'Train Absolute Error', 'Validation Absolute Error'])
        fig.tight_layout()
        #plt.figure(figsize=(14, 24))
        fig.savefig(save_img_path + '/learning_curve.png', bbox_inches='tight')

class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.

    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.

    Pixel accuracy measures how many pixels in a image are predicted correctly.

    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.

    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.

    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.

    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.

    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5

        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.

        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.

        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.

    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.

    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """

    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)

        return pixel_acc, dice, precision, recall

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        pixel_acc, dice, precision, recall = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return pixel_acc, dice, precision, recall


class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative
    rate, as specificity/TPR is meaningless in multiclass cases.
    """

    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        return pixel_acc, dice, precision, specificity, recall

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        pixel_acc, dice, precision, specificity, recall = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        return [pixel_acc, dice, precision, specificity, recall]

def compute_miou(x_pred, x_output):
    # evaluation metircs from https://github.com/lorenmt/mtan
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        for j in range(20):
            pred_mask = torch.eq(x_pred_label[i],
                                 j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda())
            true_mask = torch.eq(x_output_label[i],
                                 j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda())
            mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
            union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size
