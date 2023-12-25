import torch
import torch.nn as nn
import torch.nn.functional as F

from .tal import TaskAlignedAssigner
from .box_utils import make_anchors, dist2bbox, bbox2dist, bbox_iou

class CenterNetLoss(nn.Module):
    def __init__(self, ratio, pos_threshold, cls_weight, offset_weight, shape_weight, class_neg_weight) -> None:
        super().__init__()
        self.ratio = ratio
        self.pos_threshold = pos_threshold
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight
        self.shape_weight = shape_weight
        self.class_neg_weight = class_neg_weight
    
    def cls_loss(self, pred_score, target_score):
        ratio = self.ratio
        pos_threshold = self.pos_threshold
        pred = torch.sigmoid(pred_score).squeeze()
        
        pos_index = target_score.gt(pos_threshold).float()
        neg_index = target_score.lt(pos_threshold).float()

        neg_weight = torch.pow(1 - target_score, 4)
        pred = torch.clamp(pred, 1e-5, 1-1e-5)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_index
        neg_loss = self.class_neg_weight * torch.log(1 - pred) * torch.pow(pred, 2) * neg_weight * neg_index
        
        pos_num = pos_index.float().sum()
        #OHEM
        b = neg_loss.shape[0]
        neg_loss = neg_loss.flatten()
        numhard_cnt = min(len(neg_loss), int(ratio * pos_num.item() * b))
        
        if pos_num > 0:
            _, index = torch.topk(-neg_loss, numhard_cnt)
            neg_loss = torch.index_select(neg_loss, -1, index)
            
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        cls_loss = -(pos_loss + neg_loss) / pos_num if pos_num > 0 else -neg_loss
            
        return cls_loss, pos_index, pos_num
        
    def forward(self, output, targets):
        # unpack target
        target_score = torch.stack([target['hm'] for target in targets])
        target_offset = torch.stack([target['point_reg'] for target in targets])
        target_shape = torch.stack([target['rad'] for target in targets])
        loss = torch.zeros(3, device=target_score.device) # 

        pred_score, pred_offset, pred_shape = output
        
        pred_offset = pred_offset.permute(0, 2, 3, 4, 1)
        pred_shape = pred_shape.squeeze()
        # get cls loss
        loss[0], pos_mask, pos_num = self.cls_loss(pred_score, target_score)
        
        # get offset and shape loss
        if pos_num > 0:
            masked_offset = pred_offset * pos_mask.unsqueeze(-1)
            masked_shape = pred_shape * pos_mask
            
            loss[1] = F.smooth_l1_loss(masked_offset.flatten(), target_offset.flatten(), reduction="sum") / pos_num
            loss[2] = F.smooth_l1_loss(masked_shape.flatten(), target_shape.flatten(), reduction='sum') / pos_num
        
        loss[0] *= self.cls_weight
        loss[1] *= self.offset_weight
        loss[2] *= self.shape_weight 

        return loss.sum(), loss.detach()
        
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
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
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
            
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        
        anchor_points, stride_tensor = make_anchors(feat, self.crop_size, 0.5)
        gt_labels, gt_bboxes = targets.split((1, 6), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri) 
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        
        loss[0] *= self.box_weight
        loss[1] *= self.cls_weight
        loss[2] *= self.dfl_weight
        
        return loss.sum(), loss.detach()