# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
       
        if config is None or model is None or optimizer is None or logger is None: # check if all the necessary components are implemented
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.speed_up()  # 调用加速方法（将模型移动到 GPU 上）# move model to GPU
        self.timenow = time_now # get current time 由 train.py 已经创建过 这样两边使用完全相同的实验 ID

        # 初始化一个字典，用于记录所有 epoch 中最佳指标   best_metrics_all_time['test']如果不存在：自动创建：float('-inf') 
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf') #lambda : 100 没有输入参数：输出100 把值包装成函数 defaultdict 不接受值，只接受函数
            if self.metric_scoring != 'eer' else float('inf'))
        )

        # create directory path # 如果没有特定任务目标，直接用模型名称和时间戳命名目录   # 如果有特定任务目标，则将其加入目录名称
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'], # 日志根目录
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],   # 日志根目录
                self.config['model_name'] + task_str + '_' + self.timenow   # 模型名称 + 任务目标 + 时间戳
            )
        os.makedirs(self.log_dir, exist_ok=True)  # 创建日志目录，如果目录已存在则忽略

    def get_writer(self, phase, dataset_key, metric_key):   #你告诉我： 阶段 + 数据集 + 指标  我给你： 对应的 TensorBoard 写入器 
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:  #这里采用的是懒创建。判断这个 TensorBoard 写入器有没有创建过
            writer_path = os.path.join(  
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True) # 创建目录，exist_ok=True如果目录已经存在，不报错，直接跳过。
            self.writers[writer_key] = SummaryWriter(writer_path) #创建真正的 TensorBoard writer 深度学习工程里非常典型的：用字典管理大量对象实例。
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device) #真正把模型搬到 GPU
        self.model.device = device  #它只是给 Python 对象增加一个属性，告诉这个模型对象：我的设备在哪里？"但是它不会移动任何参数
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count() #计数本台机器的GPU数量
            print(f'avai gpus: {num_gpus}')
            # self.config['local_rank'] = [i for i in range(0,num_gpus)] #  调试代码 列表推导式 range 本身不是列表，它是一个可迭代对象 
            # print(self.config['local_rank'])
            self.model = DDP(
                self.model, device_ids=[self.config['local_rank']],
                find_unused_parameters=True,  #  允许：某一次 forward 中，有些参数没有参与 loss 的计算。
                output_device=self.config['local_rank']
                )
            #就是给模型外面套了一层 DDP 包装器： self.model.module才是原始模型。

    def setTrain(self): # 真正影响的是例如：Dropout BatchNorm
        self.model.train()                                                                                                   =
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path): # 磁盘 checkpoint恢复到 self.model
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu') #把 checkpoint 先加载到 CPU。保存checkpoint的GPU编号和当前机器GPU编号可能不同
            suffix = model_path.split('.')[-1]
            if suffix == 'p': 
                self.model.load_state_dict(saved.state_dict())  # 文件中保存整个 model 对象
            else:
                self.model.load_state_dict(saved)   # 文件里直接保存 state_dict
            self.logger.info('Model found in {}'.format(model_path)) 
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            self.optimizer.zero_grad()
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            # self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()
            return losses,predictions


    def train_epoch( 
        self,
        epoch,    # 当前的训练轮次
        train_data_loader,    # 训练数据加载器  len(train_dataset)返回的是一个epoch包含多少个batch，即要迭代多少步
        test_data_loaders=None,  # 测试数据加载器（默认为None）
        ):  # 训练一个 epoch，并在指定的步骤上进行测试和记录。

        # 日志记录当前epoch开始的信息
        self.logger.info(f"===> Epoch[{epoch} ] start!")

        # 设置每个epoch的测试次数，如果是第一轮测试只进行一次，否则进行两次
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1
 
        test_step = len(train_data_loader) // times_per_epoch    # 计算在每个epoch中的测试间隔步数 //是取整除法
        step_cnt = epoch * len(train_data_loader)    # 计算当前epoch开始时的全局训练步数

        # 保存训练数据字典 save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))

        # 初始化用于记录损失和指标的对象 define training recorder
        train_recorder_loss = defaultdict(Recorder)    # 记录训练损失
        train_recorder_metric = defaultdict(Recorder)   # 记录训练指标

        # 遍历训练数据
        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)): 
            self.setTrain()   # 设置模型为训练模式
            # 将数据转移到 GPU more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            # 执行一个训练步骤，返回损失和预测结果
            losses,predictions=self.train_step(data_dict)

            # update learning rate
            # 如果使用了 SWA (Stochastic Weight Averaging)，更新模型参数
            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # 计算当前批次的数据指标
            # compute training metric for each batch data
            if type(self.model) is DDP:    # 如果使用了分布式数据并行
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:   # 普通单机训练
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            # 将损失和指标记录下来
            for name, value in batch_metrics.items():   # 记录指标
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():  # 记录损失
                # 关键：使用 .item() 获取标量值，切断计算图
                # 这样 Recorder 保存的只是一个 float，而不是整个计算图
                if isinstance(value, torch.Tensor):
                    train_recorder_loss[name].update(value.item())
                else:
                    train_recorder_loss[name].update(value)
            # run tensorboard to visualize the training process
            # 每 300 个迭代记录到 TensorBoard，并打印损失和指标
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()  # 更新学习率
                # info for loss
                # 打印并记录损失
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()  # 计算损失的平均值
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                # 打印并记录指标
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)


                # 清除记录器中的数据
                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()
            # 按照测试间隔进行测试
            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp']):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1   # 更新总的迭代步数
        return test_best_metric  # 返回测试的最佳指标

    # 计算ACC
    def get_respect_acc(self,prob,label):  # prob 模型预测的概率数组，，label: 真实标签数组（通常是 0 表示负类，1 表示正类）
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)   # 生成一个布尔数组 judge，其中 True 表示预测正确，False 表示预测错误
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])  # judge[zero_num:]:表示负类样本的判断结果
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])  # judge[:zero_num] 表示负类样本的判断结果
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        # feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            # feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists)  #,np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        # 将模型设置为评估模式（关闭Dropout、BatchNorm的train模式行为）
        self.setEval()

        # 初始化结果记录容器
        # define test recorder
        losses_all_datasets = {}   # 存储所有测试集的损失值（字典格式：数据集名->损失列表）
        metrics_all_datasets = {}  # 存储所有测试集的评估指标（字典格式：数据集名->指标字典）
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()  # 获取所有测试数据集的名称列表
        for key in keys:  # 逐个处理每个测试集
            # save the testing data_dict  保存当前测试集的元数据（如图像路径等）
            data_dict = test_data_loaders[key].dataset.data_dict   # 从DataLoader中提取原始数据字典
            self.save_data_dict('test', data_dict, key)   # 调用自定义方法保存数据字典（可能用于后续分析

            # compute loss for each dataset
            # 在当前测试集上进行推理并获取结果
            # test_one_dataset返回：损失记录器、预测结果numpy数组、标签numpy数组、特征numpy数组
            losses_one_dataset_recorder, predictions_nps, label_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            # 记录当前数据集的损失值
            losses_all_datasets[key] = losses_one_dataset_recorder
            # 计算当前测试集的各项评估指标（传入预测结果、真实标签和图像名称）
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            # 累加指标到平均统计（只处理预定义的指标）
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value

            # 记录当前数据集的主评估指标到dataset_dict（用于后续模型选择）
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]   # self.metric_scoping指定主指标（如AUC）

            # 特殊处理SWA模型（随机权重平均模型只需记录指标，不参与最佳模型保存）
            if type(self.model) is AveragedModel:  # 判断是否为SWA模型
                # 构建指标日志字符串
                metric_str = f"Iter Final for SWA:    "  # SWA模型的最终迭代标识
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "  # 拼接所有指标值
                self.logger.info(metric_str)  # 输出日志
                continue  # 跳过后续模型保存步骤
            # 保存当前数据集的最佳模型（如果当前指标优于历史最佳）
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)
        # 当配置开启保存平均指标时（需所有数据集处理完成后执行）
        if len(keys)>0 and self.config.get('save_avg',False):  # 确保至少有一个数据集且配置允许
            # calculate avg value  # 计算各指标的平均值（总累计值/数据集数量)
            for key in avg_metric:
                if key != 'dataset_dict':  # 排除存储详细指标的字段
                    avg_metric[key] /= len(keys)
            # 将平均指标视为一个虚拟数据集进行保存（数据集名为'avg'）
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
