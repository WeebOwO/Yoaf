# system related module
import os
import setproctitle
from argparse import ArgumentParser
# torch related module
import torch
import torch.backends.cudnn as cudnn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
# my module
from config import config
from models import Model, tood_build_config, basic_build_config
from datasets import LunaDataModule
from utils.loss import DetectionLoss
from utils.misc import make_folder
# global zone 
L.seed_everything(seed=42)

def main(hparams):
    # set cudnn to accelerate our model 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    # define our model
    model = Model(
        build_dict=basic_build_config(),
        train_config=dict(
            lr=hparams.init_lr,
            momentum=hparams.momentum,
            warm_up=hparams.warm_up, 
            t_max=hparams.epochs-hparams.warm_up
        ),
    )
    
    model.detector.set_criterion(DetectionLoss(model.detector, device=device, crop_size=config['crop_size'], cls_weight=1.0, box_weight=2.5, dfl_weight=1.5)) # loss need know some model para
    # define train config
    luna_data_module = LunaDataModule(json_dir=f"spilts/split_fold{hparams.fold}.json", 
                                      data_dir=config['data_dir'], 
                                      num_workers=hparams.num_workers,
                                      batch_size=hparams.batch_size)

    luna_data_module.setup('fit')
    
    experiment_dir = config['experiment_dir']      
    log_dir = f"tb_logs/fold{hparams.fold}"
    model_dir = os.path.join(experiment_dir, f"fold{hparams.fold}")
    make_folder([experiment_dir, log_dir, model_dir])
    
    logger = TensorBoardLogger(log_dir, name="my_model")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, save_weights_only=True, filename='best')
    
    trainer = L.Trainer(
        default_root_dir=experiment_dir,
        max_epochs=hparams.epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],
    )
        
    trainer.fit(model, datamodule=luna_data_module)
    trainer.save_checkpoint(filepath=os.path.join(model_dir, "last.ckpt"), weights_only=True)

if __name__ == "__main__":
    # parse argument
    parser = ArgumentParser()

    parser.add_argument('--epochs', default=config['epochs'], type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=config['batch_size'], type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--ckpt', default=None, type=str, metavar='CKPT',
                        help='checkpoint to use')
    parser.add_argument('--init-lr', default=config['init_lr'], type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=config['momentum'], type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=config['weight_decay'], type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epoch-save', default=config['epoch_save'], type=int, metavar='S',
                        help='save frequency')
    parser.add_argument('--num-workers', default=config['num_workers'], type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--warm-up', default=config['warm_up'], type=int, metavar='OUT',
                        help='epochs for warm up')
    parser.add_argument('--fold', default=config['fold_num'], type=int, metavar='F',
                        help='current_fold')    
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='F',
                        help='gpu')
    parser.add_argument('--gpu_id', default=config['gpu_id'], type=str, metavar='F',
                        help='chosen gpu id')
    parser.add_argument('--debug', default=True, type=bool, metavar='D',
                        help='use debug mode')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    setproctitle.setproctitle(f"FCOS-Fold-{args.fold}")
    
    main(args)