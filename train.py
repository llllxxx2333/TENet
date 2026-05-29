import os
import torch
import torch.nn.functional as F
import timm
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
import pytorch_losses

cudnn.benchmark = True
cudnn.enabled = True


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

def Hybrid_Loss(pred, target, reduction='mean'):

    pred = torch.sigmoid(pred)

    #BCE LOSS
    bce_loss = nn.BCELoss()
    bce_out = bce_loss(pred, target)

    #IOU LOSS
    iou_loss = pytorch_losses.IOU(reduction=reduction)
    iou_out = iou_loss(pred, target)

    #SSIM LOSS
    ssim_loss = pytorch_losses.SSIM(window_size=11)
    ssim_out = ssim_loss(pred, target)

    losses = bce_out + iou_out + ssim_out

    return  losses

# build the model
from LNet import LNet
model = LNet()
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0)

# set the path
train_dataset_path = opt.train_root
val_dataset_path = opt.val_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
from dataset import get_loader, test_dataset
image_root = train_dataset_path + '/V/'
depth_root = train_dataset_path + '/D/'
ti_root = train_dataset_path + '/T/'
gt_root = train_dataset_path + '/GT/'

val_image_root = val_dataset_path + '/V/'
val_depth_root = val_dataset_path + '/D/'
val_ti_root = val_dataset_path + '/T/'
val_gt_root = val_dataset_path + '/GT/'

train_loader = get_loader(image_root, gt_root, depth_root,ti_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root,val_depth_root,val_ti_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Model:")
logging.info(model)

logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
import torch.nn as nn

step = 0
best_mae = 1
best_epoch = 0

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# train function
def train(train_loader, model, optimizer, epoch, save_path,scheduler):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths,tis) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            depths = depths.cuda()
            tis = tis.cuda()
            gts = gts.cuda()
            depths = torch.cat((depths, depths, depths), dim=1)
            out = model(images,depths, tis)

            loss1 = Hybrid_Loss(out[0], gts)
            loss2 = Hybrid_Loss(out[1], gts)
            loss3 = Hybrid_Loss(out[2], gts)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()
            if i==1:
                scheduler.step()
            step = step + 1
            epoch_step = epoch_step + 1
            loss_all = loss.item() + loss_all
            if i % 10 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Lr:{:8f}'
                      .format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(),get_lr(optimizer)))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                             . format(epoch, opt.epoch, i, total_step, loss.item()))


        loss_all /= epoch_step

        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth,ti, name = test_loader.load_data()
            gt = gt.cuda()
            image = image.cuda()
            depth = depth.cuda()
            ti = ti.cuda()
            depth = torch.cat((depth, depth, depth), dim=1)

            res = model(image,depth,ti)
            res = torch.sigmoid(res)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = torch.sum(torch.abs(res - gt)) * 1.0 / (torch.numel(gt))
            mae_sum = mae_train.item() + mae_sum
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        train(train_loader, model, optimizer,epoch, save_path,scheduler=schedule)
        test(test_loader, model, epoch, save_path)
