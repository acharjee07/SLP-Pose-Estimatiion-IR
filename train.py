import torch
import argparse
import matplotlib.pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from configs import config
from lib.models import Respose
from bin.pytorch.train_utils import LitPose




parser = argparse.ArgumentParser(description='Train model for Pose Estimation')

parser.add_argument('--phase',  
                    metavar='Training Phase', 
                    type=int, 
                    choices=[1, 2], 
                    required=True,
                    help='1 for phase 1 training, 2 for phase 2 training')


args = parser.parse_args()




import wandb
wandb.login(key='eacc6a3540569d3d2f6906ea0cb93ae518bcdf29')  ##logging in wild red



class Config:
    th=.5
    seed=42
    n_epoch=100
    #### schedular###
    lr = 1e-4
    max_lr = .9e-3
    pct_start = 0.3
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    ######
    betas=(0.9, 0.999)
    eps=1e-08
    weight_decay=0.01
    amsgrad=True
    steps_per_epoch=241
seed_everything(Config.seed)


data_config={
'input_size': (3, 440, 440),
 'interpolation': 'bicubic',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'crop_pct': 1.0,
 'hm_size':(112,112)

 }





model_respose = Respose(config)

lit_model = LitPose(
    plConfig=Config,
    data_config=data_config,
    model=model_respose,
    phase=args.phase
    )
logger= WandbLogger(name='aurick pc',project='Final Experiment')  
checkpoint_callback=ModelCheckpoint(monitor='valid_auc',
                                   save_top_k=5,
                                   save_last=True,
                                   save_weights_only=False,
                                   filename='{epoch:02d}-{valid_auc:.4f}-{valid_acc:.4f}-{train_loss:.4f}-{train_acc:.4f}',
                                    verbose=False,
                                    mode='max',
                                    dirpath='./weights/p1_weights' if args.phase==1 else './weights/p2_weights'
                                   )




trainer = Trainer(auto_lr_find=Config.lr,
    max_epochs=Config.n_epoch,
    gpus=[0],
    callbacks=checkpoint_callback,

    logger=logger,
    weights_summary='top',
    amp_backend='native'
)

trainer.fit(lit_model)