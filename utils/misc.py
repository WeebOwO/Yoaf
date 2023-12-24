import os
import torch
import numpy as np
import torch.distributions as dist
from torch.distributions import MultivariateNormal

def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxelToWorldCoord(voxelCoord, origin, spacing):
    worldCoord = voxelCoord * spacing
    worldCoord += origin
    return worldCoord

def kl_divergence(mu1, cov1, mu2, cov2):
    """
    Calculate KL Divergence between two multivariate Gaussian distributions.
    mu1, mu2: Mean vectors of the two distributions.
    cov1, cov2: Covariance matrices of the two distributions.
    """
    dist1 = MultivariateNormal(mu1, cov1)
    dist2 = MultivariateNormal(mu2, cov2)

    # Calculate KL Divergence
    kl_div = dist.kl.kl_divergence(dist1, dist2).sum()

    return kl_div.item()

def make_folder(file_paths):
    for path in file_paths:
        if not os.path.exists(path):
            os.mkdir(path)
            
def collate_v8(batches):
    batch = []
    [batch.extend(b) for b in batches]
    bs = len(batch)
    imgs = [s['image'] for s in batch]
    annots = [s['box'] for s in batch]

    imgs = np.stack(imgs)
    # current box foramt is zyxzyx
    max_obj_count = max(annot.shape[0] for annot in annots)
    
    annot_padded = -1 * np.ones((bs, max_obj_count, 7), dtype='float32')

    if max_obj_count > 0:
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], 1:] = annot
                annot_padded[idx, :annot.shape[0], 0] = 0
    else:
        annot_padded = -1 * np.ones((len(annots), 1, 7), dtype='float32') 

    return {'image': torch.tensor(imgs), 'targets': torch.tensor(annot_padded)} # this is return target

def collate_dict(batch):
    inputs = torch.stack([batch_data_ii["image"] for batch_data_i in batch for batch_data_ii in batch_data_i])
    targets = [
        dict(
            hm=batch_data_ii['hm_label'],
            point_reg=batch_data_ii['point_reg_label'],
            rad=batch_data_ii['rad']
        )
        for batch_data_i in batch
        for batch_data_ii in batch_data_i
    ]
        
    return {"image" : inputs, "targets" : targets}

def build_norm(norm_cfg):
    return 