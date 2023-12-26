import torch
import torch.nn as nn
import torch.nn.functional as F

from .tal import TaskAlignedAssigner
from .box_utils import make_anchors, dist2bbox, bbox2dist, bbox_iou
        
class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

class DetectionLoss(nn.Module):
    def __init__(self, model, device, crop_size, cls_weight=0.5, box_weight=7.5, dfl_weight=1.5) -> None:
        super().__init__()
     
        self.reg_max = model.head.reg_max
        self.nc = model.head.cls
        self.use_dfl = self.reg_max  > 1
        
        self.device = device
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(self.reg_max, dtype=torch.float).to(device)

        self.crop_size = crop_size
        self.cls_loss = FocalLoss()
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        
    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 6, c // 6).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
        
    def forward(self, preds, targets):
        loss = torch.zeros(3, device=self.device) 
        feat = preds[1]
        
        pred_scores = preds[0].view(feat.shape[0], 1, -1).permute(0, 2, 1).contiguous()
        pred_distri = preds[1].view(feat.shape[0], self.reg_max * 6, -1).permute(0, 2, 1).contiguous()
                    
        anchor_points, stride_tensor = make_anchors(feat, self.crop_size, 0.5)
        gt_labels, gt_bboxes = targets.split((1, 6), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri) 
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        loss[1] = self.cls_loss(pred_scores, target_scores) / target_scores_sum 
        
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        
        loss[0] *= self.box_weight
        loss[1] *= self.cls_weight
        loss[2] *= self.dfl_weight
        
        return loss.sum(), loss.detach()