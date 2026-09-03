"""
eval pretained model.
"""
import os
import json  # 用于保存可复核的逐数据集测试指标
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
from path_config import TRAINING_ROOT, resolve_data_paths

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default=str(TRAINING_ROOT / 'config/detector/lora.yaml'),       # 检测器路径
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+",default=["FaceForensics++"])
parser.add_argument('--weights_path', type=str,
                    default=None)    # 权重路径
parser.add_argument('--output_dir', type=str, default=None, help='directory for evaluation reports')  # 指定持久化测试报告目录
parser.add_argument('--no-save_tsne', dest='save_tsne', action='store_false', default=None, help='disable t-SNE feature export')  # 评测时可关闭非必要特征导出
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
    total_batches = len(data_loader)  # 固定总 batch 数，用于输出可追踪的普通文本进度

    for i, data_dict in tqdm(enumerate(data_loader), total=total_batches, disable=True):  # 始终禁用光标控制进度条，避免 tmux 和日志出现 ANSI 控制符
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
        if (i + 1) % 50 == 0 or i + 1 == total_batches:
            print(f"progress: {i + 1}/{total_batches}", flush=True)  # 每 50 个 batch 输出一行稳定进度，最后一个 batch 也输出
    
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
        print(f"dataset: {key}", flush=True)  # 在推理前立即标记当前测试集，避免长时间无输出
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


def save_metrics_report(metrics_all_datasets, output_dir, config, weights_path):
    scalar_metrics = {name: {key: float(value) for key, value in metrics.items() if key not in {'pred', 'label'}} for name, metrics in metrics_all_datasets.items()}  # 去除逐样本数组以生成紧凑 JSON
    frame_datasets = ['Celeb-DF-v1', 'Celeb-DF-v2', 'DeepFakeDetection', 'DFDC', 'DFDCP']  # 论文五集 frame AUC 的固定顺序
    video_datasets = ['Celeb-DF-v2', 'DeepFakeDetection', 'DFDC', 'DFDCP']  # 论文四集 video AUC 的固定顺序
    frame_average = float(np.mean([scalar_metrics[name]['auc'] for name in frame_datasets]))  # 计算五集 frame AUC 平均值
    video_average = float(np.mean([scalar_metrics[name]['video_auc'] for name in video_datasets]))  # 计算四集 video AUC 平均值
    results = {'checkpoint': weights_path, 'data_root': config['data_root'], 'datasets': scalar_metrics, 'paper_summary': {'frame_auc_5set': frame_average, 'video_auc_4set': video_average}}  # 汇总可复核的评测元数据与指标
    os.makedirs(output_dir, exist_ok=True)  # 创建独立结果目录，避免覆盖训练日志
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as file:  # 以机器可读格式保存逐集指标
        json.dump(results, file, indent=2, ensure_ascii=False)  # 保留完整浮点精度与中文可读性
    with open(os.path.join(output_dir, 'evaluation.log'), 'w', encoding='utf-8') as file:  # 以论文阅读友好的文本格式保存指标
        file.write(f"Checkpoint: {weights_path}\nData root: {config['data_root']}\n\n")  # 记录权重和本地数据来源
        file.write("Frame AUC (5 sets)\n")  # 写入论文主表的 frame 指标段落
        for name in frame_datasets:  # 按论文顺序写入五个数据集 AUC
            file.write(f"{name}: {scalar_metrics[name]['auc']:.6f}\n")  # 写入单数据集 frame AUC
        file.write(f"Frame Avg (5 sets): {frame_average:.6f}\n\n")  # 写入五集 frame AUC 平均值
        file.write("Video AUC (4 sets)\n")  # 写入论文主表的 video 指标段落
        for name in video_datasets:  # 按论文顺序写入四个数据集 video AUC
            file.write(f"{name}: {scalar_metrics[name]['video_auc']:.6f}\n")  # 写入单数据集 video AUC
        file.write(f"Video Avg (4 sets): {video_average:.6f}\n\n")  # 写入四集 video AUC 平均值
        file.write("All scalar metrics\n")  # 附加保存所有可用评测标量，供论文复核
        for name, metrics in scalar_metrics.items():  # 逐数据集输出 acc、auc、eer、ap 和 video_auc
            file.write(f"{name}: {metrics}\n")  # 保留完整单数据集标量字典
    return results


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(TRAINING_ROOT / 'config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    resolve_data_paths(config)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    if args.save_tsne is not None:
        config['save_tsne'] = args.save_tsne  # 命令行显式覆盖配置中的特征导出开关
    
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

    output_dir = args.output_dir or os.path.join('logs', 'testing', f"{config['model_name']}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")  # 默认使用时间戳目录保存独立报告
    save_metrics_report(best_metric, output_dir, config, weights_path)  # 保存论文五集汇总和全部逐集指标
    print(f'===> Results saved to {output_dir}')  # 输出结果目录供 tmux/console 日志直接定位

    # save tsne
    fixed_save_path = str(TRAINING_ROOT / "tsne/our_diff.pkl")

    if config['save_tsne']:
        # (可选，但强烈推荐): 确保目录存在，如果不存在就创建它
        os.makedirs(os.path.dirname(fixed_save_path), exist_ok=True)

        with open(fixed_save_path, 'wb') as f:
            pickle.dump(dict(tsne_dict), f)
        print('===> Save tsne done!')


if __name__ == '__main__':
    main()
