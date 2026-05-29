import torch
import torch.nn.functional as F

import sys
sys.path.append('./models')
import numpy as np
import os
import cv2
from dataset import test_dataset
from LNet import LNet
from config import opt

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=False):
    import copy
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = LNet()
repvgg_model_convert(model)
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load('./save_path2/test_best.pth'))
model.cuda()
model.eval()


#test
test_mae = []
test_datasets = ['Test']

from dataset import test_dataset
for dataset in test_datasets:
    mae_sum  = 0
    savepath = './savepath2/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    image_root = dataset_path + dataset + '/V/'
    depth_root = dataset_path + dataset + '/D/'
    gt_root = dataset_path + dataset + '/GT/'
    ti_root=dataset_path + dataset +'/T/'
    test_loader = test_dataset(image_root, gt_root, depth_root,ti_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, depth,ti, name  = test_loader.load_data()
        gt = gt.cuda()
        image = image.cuda()
        depth = depth.cuda()
        ti = ti.cuda()
        depth = torch.cat((depth,depth,depth),dim=1)
        res = model(image,depth,ti)
        predict = torch.sigmoid(res)
        predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
        mae = torch.sum(torch.abs(predict - gt)) / torch.numel(gt)
        # mae = torch.abs(predict - gt).mean()
        mae_sum = mae.item() + mae_sum
        predict = predict.data.cpu().numpy().squeeze()
        # print(predict.shape)
        print('save img to: ',savepath+name)
        cv2.imwrite(savepath+name, predict*255)
    test_mae.append(mae_sum / test_loader.size)
print('Test Done!', 'MAE', test_mae)
