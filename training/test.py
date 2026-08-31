"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import itertools

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict
# "FaceForensics++","FF-F2F", "FF-DF", "FF-FS", "FF-NT", "Celeb-DF-v1","Celeb-DF-v2","DeepFakeDetection","FaceShifter", "DFDCP", "UADFV"

# ["VQGAN_ff","StyleGAN2_ff","StyleGAN3_ff","StyleGANXL_ff","sd2.1_ff","ddim_ff","rddm_ff","pixart_ff","DiT_ff","SiT_ff","whichisreal"]
import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/detector/lora.yaml',       # 检测器路径
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+",default=["FaceForensics++"])
parser.add_argument('--weights_path', type=str,
                    default='/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-30-00-23-18/test/avg/ckpt_best.pth')    # 权重路径
# /root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-28-14-19-40/test/avg/ckpt_best.pth  lora
# /root/csy-7pw03c/disk/project/DeepfakeBench-main/training/logs/training/lora_2025-11-30-00-23-18/test/avg/ckpt_best.pth   lora+vae  - lora.yaml
# /home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/our_noise_0.2/test/avg/ckpt_best.pth
# /home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/weights/spsl_best.pth
# /home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/aour1_2025-06-09-12-18-23/test/avg/ckpt_best.pth
# /home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/weights/ucf_best.pth
# /home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/lsda_2025-03-20-21-55-15/test/avg/lsda.pth
# /home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/logs/training/sbi_2024-12-09-16-43-28/test/avg/ckpt_best.pth  sbi


#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    spe_label_list = []

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        if model.config['save_tsne']:
            feature_lists += list(predictions['feat_diff'].cpu().detach().numpy())  # 差分特征：仅用于提供feat_diff的Detector
            # feature_lists += list(predictions['feat'].cpu().detach().numpy())  # 通用特征：注释上一行并取消本行注释即可切换
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    tsne_dict = defaultdict(lambda: defaultdict(list))

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps,feat_nps = test_one_dataset(model, test_data_loaders[key])

        # 读取 spe_label 和 feat
        if model.config['save_tsne']:
            tsne_dict[key]['feat'].append(feat_nps)
            tsne_dict[key]['label'].append(label_nps)
            tsne_dict[key]['pred'].append(predictions_nps)
            tsne_dict[key]['spe_label'].append(test_data_loaders[key].dataset.data_dict['spe_label'])
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")


        if model.config['save_tsne']:
            # print(f"before concat, feat shape is: {tsne_dict['feat'][0].shape}, label is: {tsne_dict['label']}")
            tsne_dict[key]['feat'] = np.concatenate(tsne_dict[key]['feat'], axis=0)
            tsne_dict[key]['label'] = np.concatenate(tsne_dict[key]['label'], axis=0)
            tsne_dict[key]['pred'] = np.concatenate(tsne_dict[key]['pred'], axis=0)
            tsne_dict[key]['spe_label'] = list(itertools.chain(*tsne_dict[key]['spe_label']))

    return metrics_all_datasets,tsne_dict

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')

    # start testing
    best_metric, tsne_dict = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

    # save tsne
    fixed_save_path = "/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/tsne/our_diff.pkl"

    if config['save_tsne']:
        # (可选，但强烈推荐): 确保目录存在，如果不存在就创建它
        os.makedirs(os.path.dirname(fixed_save_path), exist_ok=True)

        with open(fixed_save_path, 'wb') as f:
            pickle.dump(dict(tsne_dict), f)
        print('===> Save tsne done!')


if __name__ == '__main__':
    main()

