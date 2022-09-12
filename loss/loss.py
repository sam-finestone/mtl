import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils import ramps

__all__ = ['SegmentationLosses', 'OhemCELoss2D', 'AdvLoss', 'DiceLoss', 'DiceBCELoss']

# class SegmentationLosses(nn.CrossEntropyLoss):
#     """2D Cross Entropy Loss with Auxilary Loss"""
#     def __init__(self,
#                  weight=None,
#                  ignore_index=-1):
#
#         super(SegmentationLosses, self).__init__(weight, None, ignore_index)
#
#     def forward(self, pred, target):
#         return super(SegmentationLosses, self).forward(pred, target)
#
# class OhemCELoss2D(nn.CrossEntropyLoss):
#     """2D Cross Entropy Loss with Auxilary Loss"""
#     def __init__(self,
#                  n_min,
#                  thresh=0.7,
#                  ignore_index=-1):
#
#         super(OhemCELoss2D, self).__init__(None, None, ignore_index, reduction='none')
#
#         self.thresh = -math.log(thresh)
#         self.n_min = n_min
#         self.ignore_index = ignore_index
#
#     def forward(self, pred, target):
#         return self.OhemCELoss(pred, target)
#
#     def OhemCELoss(self, logits, labels):
#         loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
#         loss, _ = torch.sort(loss, descending=True)
#         if loss[self.n_min] > self.thresh:
#             loss = loss[loss>self.thresh]
#         else:
#             loss = loss[:self.n_min]
#         return torch.mean(loss)

# Depth inverse depth l1 loss
# loss fucntion taken from MTAN code base  - https://github.com/lorenmt/mtan/blob/master/im2im_pred/utils.py
class InverseDepthL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_pred, x_output):
        device = x_pred.device
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask,
                                                                                     as_tuple=False).size(0)
        return loss

class InverseDepthL1Loss2(nn.Module):
    """
    L1 loss on inverse depth map ignoring -1 values
    """

    def __init__(self, invalid_idx=-1):
        self.invalid_idx = invalid_idx
        super(InverseDepthL1Loss2, self).__init__()

    def forward(self, prediction, target):
        binary_mask = (torch.sum(prediction, dim=1) != self.invalid_idx).type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = torch.sum(torch.abs(prediction - target) * binary_mask) / torch.nonzero(binary_mask).size(0)
        return loss

class L1LossIgnoredRegion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        # valid_mask = (gt != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
               / torch.nonzero(valid_mask, as_tuple=False).size(0)
        return loss

# Taken from the CCT code base - for consistency loss
class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """
    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup

def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    # assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size() # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean') # take the mean over the batch_size

def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')

def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5

# Mila  class for model training
class AdvLoss(nn.Module):
    '''Class to record dice loss for image segmentation given tensors of target and predicted masks.'''
    def __init__(self):
        super().__init__()

    def forward(self, mask_preds, mask_targets, eps=1):
        mask_preds = torch.sigmoid(mask_preds)
        # Flatten predictions and targets
        mask_preds = mask_preds.view(-1)
        mask_targets = mask_targets.view(-1)
        # Intersection
        inter = (mask_preds * mask_targets).sum()
        # Dice coefficient including eps to avoid division by 0
        dice = ((inter * 2.0) + eps) / (mask_preds.sum() + mask_targets.sum() + eps)
        dice_loss = 1 - dice
        return dice_loss

# Dice loss class for model training
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs.to(torch.float32), targets.to(torch.float32), reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def model_fit(x_pred, x_output, task_type):
    device = x_pred.device
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)
    loss = float('inf')
    if task_type == 'segmentation':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(
            0)
    return loss


# class Discriminator(nn.Module):
#     # self.model = nn.Sequential(series of convs) - outputs a binary probability of being from Slow or Fast
#     # self.act = nn.Sigmoid()
#
#     def forward(self, x_slow, x_fast):
#         return self.act(self.model(x_slow)), self.act(self.model(x_fast))
#
#
# def loss_function(y_pred, y_gt, x_slow_all, x_fast_all, D=Discriminator()):
#     # prediction loss
#     loss_1 = nn.CrossEntropyLoss()(y_pred, y_gt)
#
#     # adversarial loss
#     loss_l1 = nn.L1Loss()(x_slow_all, x_fast_all)
#     slow_prob = D(x_slow_all)
#     fast_prob = D(x_fast_all)
#     loss_adv = nn.BCELoss()
#     labels = torch.tensor(torch.ones_like(slow_prob), torch.zeros_like(fast_prob))
#     loss_adv_c = loss_adv([slow_prob, fast_prob], labels)
#     loss_from_paper = alpha * loss_l1 - beta * loss_adv_c
#
#     return loss_1, loss_from_paper