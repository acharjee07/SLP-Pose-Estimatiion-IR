import torch
import json
import argparse
import numpy as np
from lib.models import Respose
from configs import config
from bin.pytorch.train_utils import get_preds
from bin.pytorch.train_utils import LitPose
from bin.pytorch.inference_utils import get_loader,get_results

train_path='./data/train/train/'
test_path='./data/test1/'
valid_path='./data/valid/valid/'


parser = argparse.ArgumentParser(description='Inference for Pose Estimation')

parser.add_argument('--checkpoint',  
                    nargs='+',
                    metavar='Previously saved training checkpoint', 
                    type=str, 
                    required=True,
                    help='path for training weight (checkpoint')


args = parser.parse_args()



def adjustKeyPts5771(keyPts, heatmaps):

    keyPtsFloat = np.float32(keyPts)
    keyPtsFloat[..., 0] =112.0 * (keyPts[..., 0] / 120.0)
    keyPtsFloat[..., 1] = 112.0 * (keyPts[..., 1] / 160.0)

    chunkWindows = [5, 5, 3, 3, 3]

    for idx in range(keyPtsFloat.shape[0]):
        for jdx in range(keyPtsFloat.shape[1]):
            hm = heatmaps[idx, jdx]

            for w in chunkWindows:
                px, py = np.int16(keyPtsFloat[idx, jdx])
                if px < hm.shape[1] - (w + 1) and px > w and py < hm.shape[0] - (w + 1) and py > w:
                    hmChunk = hm[py - w:py + w + 1, px - w:px + w + 1]
                    hmMass = np.stack([hmChunk, hmChunk], axis=-1)
                    hmGrid = np.stack(np.meshgrid(np.arange(-w, w + 1),
                                                  np.arange(-w, w + 1)), axis=-1)

                    hmCOM = np.mean(np.mean(hmGrid * hmMass, axis=0), axis=0)
                    keyPtsFloat[idx, jdx] += hmCOM

    keyPtsFloat[..., 0] = 120.0 * (keyPts[..., 0] / 112.0)
    keyPtsFloat[..., 1] = 160.0 * (keyPts[..., 1] / 112.0)

    return keyPtsFloat

class Config:
    th=.5
    seed=42
    n_epoch=100
    #### schedular ###
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
                phase=0)

checkpoint = args.checkpoint

test_img_paths,flipped_loader_test = get_loader(
                                                path=test_path,
                                                config=data_config,
                                                loader_type='test',
                                                flip=True)
_, notflipped_loader_test=get_loader(
                                    path=test_path,
                                    config=data_config,
                                    loader_type='test',
                                    flip=False
                                    )
result_test1=get_results(
                        model=lit_model,
                        checkpoint=checkpoint[0],
                        prediction_type='test',
                        loadern=notflipped_loader_test,
                        loaderf=flipped_loader_test
                        )
result_test2=get_results(
                        model=lit_model,
                        checkpoint=checkpoint[1],
                        prediction_type='test',
                        loadern=notflipped_loader_test,
                        loaderf=flipped_loader_test
                        )






hm1=result_test1[3]
hm2=result_test2[3]
paths=test_img_paths

hm_f=[]
cordsf=[]
ensambled_hm=[]
for h1,h2 in zip(hm1,hm2):
    
    hm=h1+h2
    
#     ensambled_hm.append(np.array(hm))
    cordsf.append(np.array(get_preds(hm.unsqueeze(0))[0]))
    hm_f.append(np.array(hm))
predictions={}
for pred,path in zip(cordsf,paths):
    temp=pred.copy()
    
    temp[:,0]=((temp[:,0]/data_config['hm_size'][0])*120)-1
    temp[:,1]=((temp[:,1]/data_config['hm_size'][1])*160)-1
    predictions[path]=temp
    
result_list=[]
for x in predictions:
    result_list.append(predictions[x])
    
    
test_result2=adjustKeyPts5771(np.array(result_list), np.array(hm_f))
import json
test_result2=json.dumps(np.array(test_result2).tolist())
out_file = open("./results/preds.json", "w") 
out_file.write(test_result2)
out_file.close()




# keypts=result_test[0]
# imgs=result_test[1]
# paths=test_img_paths

# predictions={}

# for pred,path in zip(keypts,paths):
#     temp=pred.clone()
    
#     temp[:,0]=((temp[:,0]/104)*120) - 1
#     temp[:,1]=((temp[:,1]/104)*160) - 1
#     predictions[path]=temp
    
# result_list=[]
# for x in predictions:
#     result_list.append(predictions[x].numpy())

# #Sharukh Prediction
# # hm=result_test[3]
# # hm_a=[]
# # for h in hm:
# #     hm_a.append(np.array(h))
# # heatmaps=np.array(hm_a)
# # keyPts=np.array(result_list)

# # result=adjustKeyPts5771(keyPts, heatmaps)

# # test_result=json.dumps(result.tolist())
# # out_file = open("./results/preds.json", "w") 
# # out_file.write(test_result)
# # out_file.close()


# #Sanjay Prediction
# # test_result=json.dumps(np.array(result_list).tolist())
# # out_file = open("./results/preds.json", "w") 
# # out_file.write(test_result)
# # out_file.close()

# #Shariar Predictions
# def update_point(pivot_point,to_be_updated,current_dist,make_dist): # shifts the point along the line according to the requirement
#     k1=current_dist.reshape(to_be_updated.shape[0],1)
#     k2=make_dist.reshape(to_be_updated.shape[0],1)
#     updated_point=(k2*to_be_updated+(k1-k2)*pivot_point)/k1
    
    
#     return updated_point

# def getKeyDists(keypoints): 
    
#     # Vector=np.zeros((14,14,2)) 
#     # Vector[:,:,0]=keypoints[:,0].reshape(1,14)-np.transpose(keypoints[:,0].reshape(1,14))
#     # Vector[:,:,1]=keypoints[:,1].reshape(1,14)-np.transpose(keypoints[:,1].reshape(1,14))
#     dist=np.linalg.norm((keypoints[:, 12, :]-keypoints[:, 13, :]), axis=1)
#     return dist

# hm=result_test[3]
# hm_a=[]
# for h in hm:
#     hm_a.append(np.array(h))
# heatmaps=np.array(hm_a)
# keyPts=np.array(result_list)
# result=adjustKeyPts5771(keyPts, heatmaps)

# # print(result.shape)
# # dist = getKeyDists(result)
# # result[:, 13, :] = update_point(result[:, 12, :],result[:, 13, :], dist,dist * 0.95)

# test_result=json.dumps(result.tolist())
# out_file = open("./results/preds.json", "w") 
# out_file.write(test_result)
# out_file.close()