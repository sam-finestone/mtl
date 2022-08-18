import torch
import math
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SegmentationLosses', 'OhemCELoss2D', 'AdvLoss', 'DiceLoss', 'DiceBCELoss']

class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self,
                 weight=None,
                 ignore_index=-1):

        super(SegmentationLosses, self).__init__(weight, None, ignore_index)

    def forward(self, pred, target):
        return super(SegmentationLosses, self).forward(pred, target)

class OhemCELoss2D(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self,
                 n_min,
                 thresh=0.7,
                 ignore_index=-1):

        super(OhemCELoss2D, self).__init__(None, None, ignore_index, reduction='none')

        self.thresh = -math.log(thresh)
        self.n_min = n_min
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return self.OhemCELoss(pred, target)

    def OhemCELoss(self, logits, labels):
        loss = super(OhemCELoss2D, self).forward(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

# Depth inverse depth l1 loss
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


class Discriminator(nn.Module):
    # self.model = nn.Sequential(series of convs) - outputs a binary probability of being from Slow or Fast
    # self.act = nn.Sigmoid()

    def forward(self, x_slow, x_fast):
        return self.act(self.model(x_slow)), self.act(self.model(x_fast))


def loss_function(y_pred, y_gt, x_slow_all, x_fast_all, D=Discriminator()):
    # prediction loss
    loss_1 = nn.CrossEntropyLoss()(y_pred, y_gt)

    # adversarial loss
    loss_l1 = nn.L1Loss()(x_slow_all, x_fast_all)
    slow_prob = D(x_slow_all)
    fast_prob = D(x_fast_all)
    loss_adv = nn.BCELoss()
    labels = torch.tensor(torch.ones_like(slow_prob), torch.zeros_like(fast_prob))
    loss_adv_c = loss_adv([slow_prob, fast_prob], labels)
    loss_from_paper = alpha * loss_l1 - beta * loss_adv_c

    return loss_1, loss_from_paper