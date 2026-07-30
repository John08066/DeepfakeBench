# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.
# 这是John添加的注释


import os
import argparse    # 用于解析命令行参数
from os.path import join
import cv2
import random
import datetime   # 处理日期和时间
import time       # 提供时间相关操作
import yaml       # 处理YAML格式的配置文件
from tqdm import tqdm     # 一个进度条工具
import numpy as np
from datetime import timedelta       # 处理时间间隔
from copy import deepcopy        # 深拷贝数据结构
from PIL import Image as pil_image     # PIL库用于图像处理

import torch
import torch.nn as nn
import torch.nn.parallel     # 用于多GPU并行训练
import torch.backends.cudnn as cudnn      # 提供对NVIDIA cuDNN的支持，加速训练
import torch.utils.data                 # 数据加载相关模块
import torch.optim as optim           # 优化器
from torch.utils.data.distributed import DistributedSampler     # 分布式采样器
import torch.distributed as dist        # 分布式训练相关模块


# 自定义模块
from optimizor.SAM import SAM    # 引入SAM优化器（Sharpness-Aware Minimization）
from optimizor.LinearLR import LinearDecayLR       # 自定义的线性学习率衰减器

from trainer.trainer import Trainer    # 训练器模块
from detectors import DETECTOR         # 检测器模块
from dataset import *                 # 数据集相关模块
from metrics.utils import parse_metric_for_print     # 工具函数，用于格式化评价指标输出
from logger import create_logger, RankFilter      # 日志记录模块wat

# 命令行参数解析器，用于接收训练相关配置
parser = argparse.ArgumentParser(description='Process some paths.') #创建一个 ArgumentParser 类型的对象
parser.add_argument('--detector_path', type=str,#它是在告诉 ArgumentParser：我的程序支持一个叫 --detector_path 的命令行选项。
                    default='/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/detector/lora.yaml', 
                    # 这里应该全部使用相对路径，方便后来人复现  ！！！！！
                    # 想要换backbone，先配置config下的detector的配置文件，然后在这里指定路径
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")    # nargs="+"支持多训练数据集 例python train.py --train_dataset FF++ CelebDF DFDC
parser.add_argument("--test_dataset", nargs="+")    # 支持多测试数据集
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True) # 是否保存模型检查点 默认保存
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True) # 是否保存特征 输入--no-save_feat 不保存
parser.add_argument("--ddp", action='store_true', default=False)     # 是否启用分布式数据并行
parser.add_argument('--local_rank', type=int, default=0)         # 分布式训练中的本地进程编号，设备编号
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
args = parser.parse_args()   # 解析命令行参数
torch.cuda.set_device(args.local_rank)     # 设置当前进程的CUDA设备


# 初始化随机种子，确保结果的可复现性
def init_seed(config):
    if config['manualSeed'] is None:    # 如果配置中没有配置随机函数，则随机生成一个种子
        config['manualSeed'] = random.randint(1, 10000)     # 随机生成一个种子
    random.seed(config['manualSeed'])    # 使每次运行都输出相同的数
    if config['cuda']:   # 如果启用CUDA，则设置CUDA随机种子
        torch.manual_seed(config['manualSeed'])  # 设置CPU随机种子
        torch.cuda.manual_seed_all(config['manualSeed']) # 设置GPU随机种子

# def set_seed(seed, use_cuda=True):
#     # seed init.
#     random.seed(seed)
#     np.random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     # torch seed init.
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.
#
#     # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
#
#     # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
#     # torch.use_deterministic_algorithms(True)

# 准备训练数据加载器，根据配置文件选择不同的数据集类
def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)   # 使用Face X-ray数据集
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)    # 使用FWA数据集
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')    # 使用SBI数据集
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')     # 使用LSDA数据集这里调用得到了所有数的路径
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else:
        train_set = DeepfakeAbstractBaseDataset(    # 默认数据集
                    config=config,
                    mode='train',
                )

    # 根据模型名称或配置决定是否使用自定义采样器或分布式采样器
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5) # 实例化一个自定义采样器，CustomSampler 可以控制如何在每个批次中选择视频和帧
        train_data_loader = \
            torch.utils.data.DataLoader( # 使用DataLoader加载数据
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler,
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:    # 如果启用分布式数据并行
        sampler = DistributedSampler(train_set)    # 使用分布式采样器
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:  # 单GPU或普通训练
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader

# 准备测试数据加载器
def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # 创建配置的副本，防止修改原始配置
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset     # 设置当前测试数据集
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(    # 默认测试数据集
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(         # 使用LRL测试数据集
                config=config,
                mode='test',
            )

        # 数据加载器
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,      # 测试时不打乱数据
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                # drop_last = (test_name=='DeepFakeDetection'),
                drop_last = False,   # 保留所有批次
            )

        return test_data_loader

    # 创建多个测试数据加载器
    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


# 选择优化器
def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer

# 选择学习率
def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:   # 如果使none，说明不需要学习率调度器
        return None
    elif config['lr_scheduler'] == 'step':      # 说明要使用StepLR调度器
        scheduler = optim.lr_scheduler.StepLR(  # 作用：每隔固定的 step_size 个 epoch，将学习率乘以一个因子 gamma
            optimizer,                       # 用于控制的优化器。
            step_size=config['lr_step'],     # 每隔多少个 epoch 调整一次学习率
            gamma=config['lr_gamma'],        # 学习率缩放因子
        )
        return scheduler  # 初始化好的 StepLR 调度器
    elif config['lr_scheduler'] == 'cosine':     # 如果配置中 lr_scheduler 被设置为 'cosine'，说明要使用CosineAnnealingLR调度器。
        scheduler = optim.lr_scheduler.CosineAnnealingLR(  # 作用：根据余弦退火曲线调整学习率，适合训练中后期逐步减小学习率的情况。
            optimizer,
            T_max=config['lr_T_max'],   # 周期的最大 epoch 数，表示完成一个周期时学习率最小
            eta_min=config['lr_eta_min'],  # 最小学习率（退火曲线的最低点）
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':   # 如果配置中 lr_scheduler 被设置为 'linear'，说明要使用LinearDecayLR调度器。
        scheduler = LinearDecayLR(     # 线性递减学习率。
            optimizer,
            config['nEpochs'],    # 总共的训练 epoch 数。
            int(config['nEpochs']/4),   # 指定从什么时候开始线性衰减学习率（可能表示最后四分之一阶段）
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))

# 选择评价指标
def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:   # 打开分类器的配置文件
        config = yaml.safe_load(f)
    with open('/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/config/train_config.yaml', 'r') as f:  # 打开训练配置文件，也就是训练集
        config2 = yaml.safe_load(f)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
     # 使用config2更新config,如果 config2 中的某个键在 config 中已经存在，则 config 中该键的值会被 config2 中对应的值替换。如果 config2 中的某个键在 config 中不存在，则会将该键值对添加到 config 中。
    config.update(config2)
    config['local_rank']=args.local_rank  # 配置训练设备
    # 如果是 dry_run 模式，设置为仅用于测试流程（不进行实际训练）
    if config['dry_run']:        # 通常表示一种“试运行”或“测试运行”模式
        config['nEpochs'] = 0      # 设置训练轮数为 0
        config['save_feat']=False  # 不保存特征数据
    # 如果从命令行提供了数据集路径参数，则覆盖配置文件中的路径设置
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    # 配置模型保存路径
    config['save_ckpt'] = args.save_ckpt
    # 配置是否保存训练特征
    config['save_feat'] = args.save_feat

    # 如果启用了 LMDB 数据集格式，设置数据集 JSON 文件路径
    if config['lmdb']:
        config['dataset_json_folder'] = '/datasets2/Deepfake/DeepfakeBench/config/dataset_json/'  # 配置训练json路径 /datasets2/Deepfake/DeepfakeBench/config/dataset_json
    # create logger
    # 创建日志文件夹并初始化日志记录器
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    task_str = f"_{config['task_target']}" if config.get('task_target', None) is not None else ""
    logger_path =  os.path.join(
                config['log_dir'],   # 日志存储目录
                config['model_name'] + task_str + '_' + timenow   # 日志文件夹名
            )
    os.makedirs(logger_path, exist_ok=True)    # 创建文件夹（若不存在则创建）
    logger = create_logger(os.path.join(logger_path, 'training.log'))   # 创建日志记录器
    logger.info('Save log to {}'.format(logger_path))    # 记录日志文件存储路径

    # 设置分布式训练参数
    config['ddp']= args.ddp
    # 打印完整的配置信息
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # 初始化随机种子，确保实验可重复性
    # init seed
    init_seed(config)
    # set_seed(1024)

     # 如果启用 cudnn 加速，设置 benchmark 模式以提升性能
    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # 如果启用分布式数据并行（DDP），初始化通信进程组
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',     # 使用 NCCL 后端进行通信（适用于 GPU）
            timeout=timedelta(minutes=30)    # 设置通信超时时间为 30 分钟
        )
        logger.addFilter(RankFilter(0))    # 仅记录主进程日志
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)   # 实例化模型

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # 初始化训练器
    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow)

    # 开始训练
    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch   # 更新模型当前训练的 epoch
        # 每个 epoch 训练并测试模型，返回最佳评估指标
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=test_data_loaders,
                )
        # 如果存在最佳评估指标，记录日志
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric)))
    # update

    # 如果模型为 'svdd' 类型，更新 R 参数
    if 'svdd' in config['model_name']:
        model.update_R(epoch)

    # 更新学习率调度器（如果有）
    if scheduler is not None:
        scheduler.step()

    # 关闭 TensorBoard 写入器（释放资源）
    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()



if __name__ == '__main__':
    main()

