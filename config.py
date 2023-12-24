import os
import torch
import random
import numpy as np

data_config = {
    # directory for putting all preprocessed results for training to this path
    'use_cpm_data' : False,
    'data_dir': "/nvmedata/fengxingyu/preprocessed",
    'cpm_dir' : "/data/CPMV2/preprocessed",
    'crop_size': (96, 96, 96),
    'experiment_dir' : 'experiment/',
    'black_ids' : ["1.3.6.1.4.1.14519.5.2.1.6279.6001.243094273518213382155770295147"]
}

net_config = {
    # Net configuration
    'chanel': 1,
    'crop_size': data_config['crop_size'],
    'infer_topk' : 32
}

loss_config = {
    "ratio" : 80,
    "reg_max" : 8,
    "pos_threshold" : 0.9,
    "cls_weight" : 1.0,
    "offset_weight" : 0.5,
    "shape_weight" : 0.5,
    "class_neg_weight" : 0.1,
} 

train_config = {
    'gpu_id' : "0",
    'fold_num' : 0,
    'rfla' : False,
    'batch_size': 6,
    'momentum': 0.9,
    'init_lr' : 1e-3,
    'weight_decay': 1e-4,
    'warm_up': 30,

    'epochs': 60,
    'epoch_save': 5,
    'num_workers': 20,

    'box_sigma' : 0.01
}

test_config = {
    'ckpt': "experiment/fold0/last.ckpt",
}

config = dict(data_config, **net_config)
config = dict(config, **train_config)
config = dict(config, **loss_config)
config = dict(config, **test_config)

