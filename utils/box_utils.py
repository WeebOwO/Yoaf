import torch
import numpy as np

def center_to_corner(boxes):
    corner_box = np.zeros((len(boxes), 6))
    for idx, box in enumerate(boxes):
        z, y, x, d = box
        corner_box[idx][0] = z - d / 2
        corner_box[idx][1] = y - d / 2
        corner_box[idx][2] = x - d / 2

        corner_box[idx][3] = z + d / 2
        corner_box[idx][4] = z + d / 2
        corner_box[idx][5] = z + d / 2
    return corner_box

def corner_to_center(boxes):
    center_box = np.zeros((len(boxes), 4), dtype=np.float32)
    for idx, box in enumerate(boxes):
        z_min, y_min, x_min, z_max, y_max, x_max = box
        center_box[idx][0] = (z_min + z_max) / 2
        center_box[idx][1] = (y_min + y_max) / 2
        center_box[idx][2] = (x_min + x_max) / 2
        center_box[idx][3] = z_max - z_min
    return center_box

def nms_3D(dets, overlap=0.5, top_k=200):
    # det {prob, ctr_z, ctr_y, ctr_x, d, h, w}
    dd, hh, ww = dets[:, 3], dets[:, 3], dets[:, 3]
    z1 = dets[:, 1] - 0.5 * dd
    y1 = dets[:, 2] - 0.5 * hh
    x1 = dets[:, 3] - 0.5 * ww
    z2 = dets[:, 1] + 0.5 * dd
    y2 = dets[:, 2] + 0.5 * hh
    x2 = dets[:, 3] + 0.5 * ww
    scores = dets[:, 0]
    areas = dd * hh * ww
    _, idx = scores.sort(0, descending=True)
    keep = []
    while idx.size(0) > 0:
        i = idx[0]
        keep.append(int(i.cpu().numpy()))
        if idx.size(0) == 1 or len(keep) == top_k:
            break
        xx1 = torch.max(x1[idx[1:]], x1[i].expand(len(idx)-1))
        yy1 = torch.max(y1[idx[1:]], y1[i].expand(len(idx)-1))
        zz1 = torch.max(z1[idx[1:]], z1[i].expand(len(idx)-1))

        xx2 = torch.min(x2[idx[1:]], x2[i].expand(len(idx)-1))
        yy2 = torch.min(y2[idx[1:]], y2[i].expand(len(idx)-1))
        zz2 = torch.min(z2[idx[1:]], z2[i].expand(len(idx)-1))

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        d = torch.clamp(zz2 - zz1, min=0.0)

        inter = w*h*d
        IoU = inter / (areas[i] + areas[idx[1:]] - inter)
        inds = IoU <= overlap
        idx = idx[1:][inds]
    
    return torch.from_numpy(np.array(keep))

def iou_3D(box1, box2):
    # need z_ctr, y_ctr, x_ctr, d
    z1 = np.maximum(box1[0] - 0.5 * box1[3], box2[0] - 0.5 * box2[3])
    y1 = np.maximum(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    x1 = np.maximum(box1[2] - 0.5 * box1[3], box2[2] - 0.5 * box2[3])

    z2 = np.minimum(box1[0] + 0.5 * box1[3], box2[0] + 0.5 * box2[3])
    y2 = np.minimum(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])
    x2 = np.minimum(box1[2] + 0.5 * box1[3], box2[2] + 0.5 * box2[3])

    w = np.maximum(x2 - x1, 0.)
    h = np.maximum(y2 - y1, 0.)
    d = np.maximum(z2 - z1, 0.)

    inters = w * h * d
    uni = box1[3] * box1[3] * box1[3] + box2[3] * box2[3] * box2[3] - inters
    uni = np.maximum(uni, 1e-8)
    ious = inters / uni
    return ious

def bbox_iou(box1, box2, DIoU=True, eps = 1e-7):
    def zyxdhw2zyxzyx(box, dim=-1):
        ctr_zyx, dhw = torch.split(box, 3, dim)
        z1y1x1 = ctr_zyx - dhw/2
        z2y2x2 = ctr_zyx + dhw/2
        return torch.cat((z1y1x1, z2y2x2), dim)  # zyxzyx bbox
    box1 = zyxdhw2zyxzyx(box1)
    box2 = zyxdhw2zyxzyx(box2)
    # Get the coordinates of bounding boxes
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = box1.chunk(6, -1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = box2.chunk(6, -1)
    w1, h1, d1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
    w2, h2, d2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0) * \
            (b1_z2.minimum(b2_z2) - b1_z1.maximum(b2_z1)).clamp(0) + eps

    # Union Area
    union = w1 * h1 * d1 + w2 * h2 * d2 - inter

    # IoU
    iou = inter / union
    if DIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        cd = b1_z2.maximum(b2_z2) - b1_z1.minimum(b2_z1)  # convex depth
        c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + 
        + (b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4  # center dist ** 2 
        return iou - rho2 / c2  # DIoU
    return iou  # IoU

def make_anchors(feat, input_size, grid_cell_offset=0):
    """Generate anchors from a feature."""
    assert feat is not None
    dtype, device = feat.dtype, feat.device
    _, _, d, h, w = feat.shape
    stride = input_size[0] / d
    sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
    sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
    sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift z
    anchor_points = torch.cartesian_prod(sz, sy, sx)
    stride_tensor = torch.full((d * h * w, 1), stride, dtype=dtype, device=device)
    return anchor_points, stride_tensor

def bbox_decode(anchor_points, pred_offsets, pred_shapes, stride_tensor, dim=-1):
    c_zyx = (anchor_points + pred_offsets) * stride_tensor
    return torch.cat((c_zyx, 2*pred_shapes), dim)  # zyxdhw bbox

def select_candidates_in_gts(anchor_points, gt_bboxes, eps=1e-9):
    n_anchors = anchor_points.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    min_corner, max_corner = gt_bboxes.view(-1, 1, 6).chunk(2, 2)
    bbox_deltas = torch.cat((anchor_points[None] - min_corner, max_corner - anchor_points[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    return bbox_deltas.amin(3).gt_(eps)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    min_cor, max_cor = distance.chunk(2, dim)
    x1y1 = anchor_points - min_cor
    x2y2 = anchor_points + max_cor
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    min_cor, max_cor = bbox.chunk(2, -1)
    return torch.cat((anchor_points - min_cor, max_cor - anchor_points), -1).clamp_(0, reg_max - 0.01)

if __name__ == "__main__":
    feature = torch.randn((1, 96, 32, 32, 32))

    gt_box = torch.tensor([[25, 25, 25, 30, 30, 30], [20, 25, 34, 30, 40, 38]]).repeat(2, 1, 1)
    
    anchor_points, _ = make_anchors(feature, (96, 96, 96), 0.5)
    mask_in_gts = select_candidates_in_gts(anchor_points, gt_box)
    
    print(mask_in_gts.sum())
    pass
    