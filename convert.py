import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from LNet import LNet
from argparse import ArgumentParser
from config import opt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

def repvgg_model_convert(model:torch.nn.Module,save_path=None,do_copy=False):
    import copy
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
            print(module)
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def convert(args):
    model = LNet()
    model.load_state_dict(torch.load(args.load))
    repvgg_model_convert(model, save_path=args.save)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--load', default="./save_path/Net_epoch_best.pth", help='path to the weights file')
    parser.add_argument('--save', default="./save_path/test_best.pth", help='path to the weights file')
    args = parser.parse_args()

    convert(args)