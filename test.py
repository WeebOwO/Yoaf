import os
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from monai.data import load_decathlon_datalist

from config import config
from models import Model, tood_build_config
from datasets import Luna16
from utils.misc import make_folder
from datasets.transforms import generate_infer_transform
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

def main(hparams):
    model = Model.load_from_checkpoint(hparams.weight, build_dict=tood_build_config())
    
    # define our dir 
    json_dir = f"spilts/split_fold{hparams.fold}.json"
    out_dir = os.path.join(config['experiment_dir'], f'fold{hparams.fold}')
    save_dir = os.path.join(out_dir, 'res')
    froc_dir = os.path.join(save_dir, 'froc')
    
    make_folder([save_dir, froc_dir])
    
    # create dataset
    inference_data = load_decathlon_datalist(
        json_dir,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=config['data_dir'],
    )
    
    dataset = Luna16(data=inference_data, transform=generate_infer_transform(), mode='eval')
    
    id_list = []
    for data in inference_data:
        filename = data['image'].split('/')[-1][:-4]
        id_list.append(filename)
    
    # infer
    model.detector.cuda()
    model.detector.eval()
    
    for i, targets in enumerate(dataset):
        input = targets['image']
        pid = id_list[i]
        print(f'-- Scan {i} pid:{pid} \n-- Predicting {input.shape}...')
        with torch.no_grad():
            input = input.cuda().unsqueeze(0)
            bbox = model.detector.inference(input)
            np.save(os.path.join(save_dir, f'{pid}_pbb.npy'), bbox[:, :5])
        
    # present froc result
    test_res = []
    for pid in id_list:
        if os.path.exists(os.path.join(save_dir, '%s_pbb.npy' % (pid))):
            bboxs = np.load(os.path.join(save_dir, '%s_pbb.npy' % (pid)))
            bboxs = bboxs[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(bboxs))
            test_res.append(np.concatenate([names, bboxs], axis=1))
            # os.remove(os.path.join(save_dir, '%s_pbb.npy' % (pid)))

    col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'probability']

    test_res = np.concatenate(test_res, axis = 0)
    test_submission_path = os.path.join(froc_dir, 'test_result.csv')
    df = pd.DataFrame(test_res, columns=col_names)
    df.to_csv(test_submission_path, index=False)

    if not os.path.exists(os.path.join(froc_dir, 'test_res')):
        os.makedirs(os.path.join(froc_dir, 'test_res'))

    noduleCADEvaluation('annos/voxel_annotations.csv',
                        'annos/voxel_annotations_exclude.csv',
                        id_list, test_submission_path, os.path.join(froc_dir, 'test_res'))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mode", type=str, default="eval",
                        help="you want to test or val")
    parser.add_argument("--weight", type=str, default=config['ckpt'],
                        help="path to model weights to be used")
    parser.add_argument('--fold', default=config['fold_num'], type=int, metavar='F',
                        help='current_fold')
    parser.add_argument('--infer-topk', default=config['infer_topk'], type=int, metavar='F',
                        help='infer_topk')
    parser.add_argument('--gpu-id', default=config['gpu_id'], type=str, metavar='F',
                        help='current_fold')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    main(args)