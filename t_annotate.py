import torch
import json
import pickle
import numpy as np
import argparse
from configs import config
from lib.models import Respose
from bin.pytorch.train_utils import LitPose
from bin.pytorch.inference_utils import get_loader,get_results
import os





parser = argparse.ArgumentParser(description='Generate annotation for Phase 2 training')

parser.add_argument('--checkpoint',  
                    metavar='Previously saved training checkpoint', 
                    type=str, 
                    required=True,
                    help='path for training weight (checkpoint')


args = parser.parse_args()

class Config:
    th=.5
    seed=42
    n_epoch=100
    #### schedular###
    lr = 1e-4
    max_lr = 1e-3
    pct_start = 0.3
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    ######
    betas=(0.9, 0.999)
    eps=1e-08
    weight_decay=0.01
    amsgrad=True
    steps_per_epoch=230


def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)

data_config={
'input_size': (3, 440, 440),
 'interpolation': 'bicubic',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'crop_pct': 1.0,
 'hm_size':(112,112)

 }



model_respose=Respose(config)
lit_model=LitPose(
                plConfig=Config,
                data_config=data_config,
                model=model_respose,
                phase=0
                )




checkpoint = args.checkpoint

# train_path='./data/train/train/'
# test_path='./data/test1/'
train_path=os.path.join('.','data','train','train')
test_path=os.path.join('.','data','test1')

unannotatedImgPaths,flipped_loader_resnet=get_loader(
                                                    path=train_path,
                                                    config=data_config,
                                                    loader_type='slp',
                                                    flip=True
                                                    )

_,notflipped_loader_resnet=get_loader(
                                    path=train_path,
                                    config=data_config,
                                    loader_type='slp',
                                    flip=False
                                    )

result_unannotated=get_results(
                                model=lit_model,
                                checkpoint=checkpoint,
                                prediction_type='slp',
                                loadern=notflipped_loader_resnet,
                                loaderf=flipped_loader_resnet
                                )



keypts=result_unannotated[0]
predictions={}

for pred,path in zip(keypts,unannotatedImgPaths):
    temp=pred.clone()
    
    temp[:,0]=(temp[:,0]/112)*120
    temp[:,1]=(temp[:,1]/112)*160
    predictions[path]=temp.numpy()

save_obj(predictions,'./results/prediciton_unannotated.pkl')