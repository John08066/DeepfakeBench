# region authorinfo

# author: Zhiyuan Yan
# email: zhiyuanyan@link.fcuhk.edu.cn
# date: 2023-03-30
# description: trainer

# endregion authorinfo

import os
import sys
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


current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)
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
        self.logger = logger #引用training/train.py里创建的 logger对象 这个 logger对象是整个项目的“总日志入口”，它会把日志写入 training.log 文件，也会打印到终端
        self.metric_scoring = metric_scoring
        self.speed_up()  # 调用加速方法（将模型移动到 GPU 上）# move model to GPU
        self.timenow = time_now # get current time 由 train.py 已经创建过 这样两边使用完全相同的实验 ID
        self.best_metrics_all_time = defaultdict(lambda: defaultdict(lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf')))# 初始化一个两层字典best_metrics_all_time[数据集][指标]，best_metrics_all_time['test']如果不存在：自动创建：float('-inf') #lambda : 100 没有输入参数：输出100 把值包装成函数 defaultdict 不接受值，只接受函数

        # create directory path # 如果没有特定任务目标，直接用模型名称和时间戳命名目录   # 如果有特定任务目标，则将其加入目录名称
        if 'task_target' not in config:
            self.log_dir = os.path.join(self.config['log_dir'], self.config['model_name'] + '_' + self.timenow)  # 日志根目录
        else: #  join() 中每个逗号分隔的是一个路径组件
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(self.config['log_dir'], self.config['model_name'] + task_str + '_' + self.timenow)  # 日志根目录 + 模型名称 + 任务目标 + 时间戳
        os.makedirs(self.log_dir, exist_ok=True)  # 创建日志目录，如果目录已存在则忽略

    def get_writer(self, phase, dataset_key, metric_key):   #你告诉我： 阶段 + 数据集 + 指标  我给你： 对应的 TensorBoard 写入器 
        writer_key = f"{phase}-{dataset_key}-{metric_key}" # 例如'test-Celeb-DF-v2-auc'
        if writer_key not in self.writers:  #这里采用的是懒创建，判断这个 TensorBoard 写入器有没有创建过
            writer_path = os.path.join(self.log_dir, phase, dataset_key, metric_key, "metric_board")
            os.makedirs(writer_path, exist_ok=True) # 创建目录，exist_ok=True如果目录已经存在，不报错，直接跳过。
            self.writers[writer_key] = SummaryWriter(writer_path) #创建真正的 TensorBoard writer 深度学习工程里非常典型的：用字典管理大量对象实例。
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device) #真正把模型权重搬到 GPU
        self.model.device = device  #它只是给 Python 对象增加一个属性，告诉这个模型对象：我的设备在哪里？"但是它不会移动任何参数
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count() #计数本台机器的GPU数量
            print(f'avai gpus: {num_gpus}')
            # self.config['local_rank'] = [i for i in range(0,num_gpus)] #  调试代码 列表推导式 range 本身不是列表，它是一个可迭代对象 
            # print(self.config['local_rank'])
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True, output_device=self.config['local_rank'])  # 允许：某一次 forward 中，有些参数没有参与 loss 的计算。就是给模型外面套了一层 DDP 包装器：self.model.module才是原始模型。

    def setTrain(self): # 真正影响的是例如：Dropout BatchNorm Dropout 每次训练时，随机把一部分神经元这一次的输出强制设为 0。下一次 forward 又重新随机 
        self.model.train()  # 不是立即开始训练。它只是告诉模型：接下来按照训练模式运行。                                                                        
        self.train = True # 只是记录一个布尔状态。这份文件中没有再使用这个变量，功能比较有限。

    def setEval(self):  # BatchNorm： train使用当前 batch 的均值和方差 eval使用训练期间累计的均值和方差  
        self.model.eval() # 仍然要注意不会自动关闭梯度。关闭梯度需要：torch.no_grad() 后面 inference() 会单独完成这件事。
        self.train = False

    # region 空接口
    # def load_ckpt(self, model_path): # 磁盘 checkpoint恢复到 self.model ！！！暂时没有调用点
    #     if os.path.isfile(model_path):
    #         saved = torch.load(model_path, map_location='cpu') #把 checkpoint 先加载到 CPU。保存checkpoint的GPU编号和当前机器GPU编号可能不同
    #         suffix = model_path.split('.')[-1]
    #         if suffix == 'p': 
    #             self.model.load_state_dict(saved.state_dict())  # 文件中保存整个 model 对象
    #         else:
    #             self.model.load_state_dict(saved)   # 默认文件里直接保存 state_dict
    #         self.logger.info(f'Model found in {model_path}') 
    #     else:
    #         raise NotImplementedError(f"=> no model found at '{model_path}'")
    # endregion 空接口

    def save_ckpt(self, phase, dataset_key,ckpt_info=None): #checkpoint 保存的是“模型当前学到的参数”
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"ckpt_best.pth")
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']: # SVDD 模型除了普通参数 θ 之外 R = 超球半径c = 超球中心
                torch.save({'R': self.model.R,'c': self.model.c,'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path) #默认if路径
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):    # SWA = Stochastic Weight Averaging
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"swa.pth")
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"feat_best.npy")
        features = fea
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_dataset_metadata(self, phase, dataset_metadata, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'dataset_metadata_{phase}.pickle') #.pickle 是 Python 用来保存序列化后的 Python 对象的一种二进制文件。
        with open(file_path, 'wb') as file:
            pickle.dump(dataset_metadata, file)  # {'image': [...], 'label': [...], 'video': [...]}
        self.logger.info(f"Dataset metadata saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle') #你可以把 Pickle 理解成：把内存里的 Python 对象“冻起来”存到硬盘，以后再原样恢复  
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)  # {'acc': 0.82, 'auc': 0.91, 'eer': 0.16, 'ap': 0.90, 'video_auc': 0.93, 'pred': [...], 'label': [...]}
        self.logger.info(f"Metrics saved to {file_path}")

   
    def train_step(self, batch_data):  # batch_data: {'image': Tensor[32,3,256,256], 'label': Tensor[32], 'mask': Tensor[32,1,256,256]}
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(batch_data) # 加载父类nn.module魔术方法__call__(),最终落到检测器类的forward() 返回 pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
                losses = (self.model.module if type(self.model) is DDP else self.model).get_losses(batch_data, predictions)  # 不同Detector可能需要完全不同的损失，Trainer不负责定义算法的loss，模型自己计算loss
                if i == 0: 
                    pred_first, losses_first = predictions, losses  # 有意区分用于报告的 loss 和用于优化的 loss
                self.optimizer.zero_grad()  # 清空旧梯度  forward → zero_grad → loss → backward → step  
                losses['overall'].backward()   # 执行后 parameter.grad 里面有梯度 ，但是并没有更新参数
                (self.optimizer.first_step if i == 0 else self.optimizer.second_step)(zero_grad=True)  # i==0: SAM 先沿最坏扰动方向移动参数 θ→θ+ϵ；i==1: 这里真正更新参数
            return losses_first, pred_first # 作者刻意返回第一次对应的是当前正常模型参数下的真实预测性能 第一次 forward： “这个学生正常考试能考多少？”  第二次 forward： “故意挑最刁钻的问题，看这个学生最差能考多少？”  
        else:
            self.optimizer.zero_grad()  # zero_grad → forward → loss → backward → step 经典 5 步曲
            predictions = self.model(batch_data)  # losses = {'overall': Tensor(0.43), 'cls': Tensor(0.38), 'mask': Tensor(0.05)}; predictions = {'cls': Tensor[32,2], 'prob': Tensor[32], 'feat': Tensor[32,2048]}
            losses = (self.model.module if type(self.model) is DDP else self.model).get_losses(batch_data, predictions)  # DDP 时取 module，否则直接用 model
            losses['overall'].backward()
            self.optimizer.step()   # image → backbone → feature → classifier → logits / probability → predictions
            return losses,predictions

    def train_epoch( 
        self,
        epoch,    # 当前的训练轮次
        train_data_loader,    # 训练数据加载器  len(train_dataset)返回的是一个epoch包含多少个batch，即要迭代多少步
        test_data_loaders=None,  # 测试数据加载器（默认为None）
        ):  # 训练一个 epoch，并在指定的步骤上进行测试和记录。

        self.logger.info(f"===> Epoch[{epoch}] start!")    # 日志记录当前epoch开始的信息
        times_per_epoch = 2 if epoch >= 1 else 1  # 设置每个epoch的测试次数，如果是第一轮测试只进行一次，否则进行两次
        test_step = len(train_data_loader) // times_per_epoch    # 计算在每个epoch中的测试间隔步数 //是取整除法
        step_cnt = epoch * len(train_data_loader)    # 计算当前epoch开始时的全局训练步数
        dataset_metadata = train_data_loader.dataset.data_dict
        self.save_dataset_metadata('train', dataset_metadata, ','.join(self.config['train_dataset'])) # ','.join(...) 是字符串 str 的方法：意思是：用 , 作为胶水，把列表中的字符串连接起来。
        train_recorder_loss = defaultdict(Recorder)    # 记录训练损失 模型forward→losses→Recorder.update()→保存整个epoch历史→average()→TensorBoard / log  
        train_recorder_metric = defaultdict(Recorder)   # 记录训练指标

        # 遍历训练数据 # iteration为局部计数，step_cnt 为全局计数
        for iteration, batch_data in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):  # batch_data 是当前一个 batch 的数据
            self.setTrain()   # 设置模型为训练模式
            for key in batch_data.keys():    # 将当前 batch 中的张量转移到 GPU
                batch_data[key] = batch_data[key].cuda() if torch.is_tensor(batch_data[key]) else batch_data[key]    # 仅对张量调用 .cuda()，自动跳过字符串（如 name）、None 等其他类型

            losses,predictions = self.train_step(batch_data)   # 执行当前 batch！！！为核心代码 这一行完成：前向传播 → 损失计算 → 反向传播 → 参数更新 → 执行完成后，模型参数已经发生改变。后面的日志和指标计算不会再更新参数
            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']: # 如果使用了 SWA (Stochastic Weight Averaging)，更新平均模型参数 
                self.swa_model.update_parameters(self.model)

            batch_metrics = self.model.module.get_train_metrics(batch_data, predictions) if type(self.model) is DDP else self.model.get_train_metrics(batch_data, predictions) # 计算当前批次的数据指标  如果使用了分布式数据并行则取 module，否则为普通单机训练
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)   # 记录指标
            for name, value in losses.items():  # 记录损失 关键：使用 .item() 获取标量值，切断计算图 这样 Recorder 保存的只是一个 float，而不是整个计算图
                train_recorder_loss[name].update(value.item() if isinstance(value, torch.Tensor) else value) #.item() 只适用于标量张量，如果是非标量张量，直接使用原始值

            if iteration % 300 == 0 and self.config['local_rank']==0:#  每 300 个batch迭代记录到 TensorBoard，并打印损失和指标
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()  # 更新学习率
                for rec_name, rec_items, tag in (
                    ('training-loss', train_recorder_loss, 'train_loss'),
                    ('training-metric', train_recorder_metric, 'train_metric'),
                ):
                    log_str = f"Iter: {step_cnt}" # 统一打印并记录损失与指标 tensorboard-loss / tensorboard-metric
                    for k, v in rec_items.items():
                        v_avg = v.average()  # 计算平均值
                        if v_avg == None:
                            log_str += f"{rec_name}, {k}: not calculated"
                            continue
                        log_str += f"\n{rec_name}--{k}: {v_avg}" # 类似 training-loss--overall: 0.45 training-loss--cls: 0.30 training-loss--aux: 0.15
                        writer = self.get_writer('train', ','.join(self.config['train_dataset']), k) # 创建'test/Celeb-DF-v2/auc' 的 TensorBoard
                        writer.add_scalar(f'{tag}/{k}', v_avg, global_step=step_cnt) # tag → 这条曲线叫什么  v_avg → Y轴：记录的指标值  step_cnt → X轴：当前训练到了第几步
                    self.logger.info(log_str) # 每次 logger.info() 会生成一条独立的日志记录，日志 Handler/Formatter 通常会在输出末尾加换行。

                for name, recorder in train_recorder_loss.items():recorder.clear()  # 清除记录器中的数据    
                for name, recorder in train_recorder_metric.items():recorder.clear() # Note we only consider the current 300 batch_size samples for computing batch-level train loss/metric
            
            if (step_cnt+1) % test_step == 0:# 按照测试间隔进行测试   #从0计数 step_cnt+1 
                if test_data_loaders is not None and (not self.config['ddp'] or dist.get_rank() == 0):  # 有测试集 且（非 DDP 或 rank 0）时才执行测试
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)
                else:  # 无测试集，或 DDP 下非 rank 0，跳过测试
                    test_best_metric = None
            step_cnt += 1   # 更新总的迭代步数，一个BATCH训练完毕，step_cnt就加1
        return test_best_metric  # 返回测试的历史最佳指标

    def get_respect_acc(self, prob, label):  # prob是模型预测为fake的概率数组，label是真实标签数组：0表示real，1表示fake
        pred = np.where(prob > 0.5, 1, 0)  # pred = [0.2, 0.8, 0.4] np.where(条件, 条件为True时取什么, 条件为False时取什么)
        judge = (pred == label)  # judge = [0, 0, 1, 1]逐个判断预测是否正确，得到与样本一一对应的布尔数组 
        real_mask = (label == 0)  # real_mask   = [True, False, True,  True, False]  所有真实样本（label=0）里面，模型预测对了多少
        fake_mask = (label == 1)  # 当 NumPy 发现 real_mask 是一个和 judge 等长的布尔数组时 True 的位置留下，False 的位置扔掉。
        acc_real = np.count_nonzero(judge[real_mask]) / np.count_nonzero(real_mask)  # 举例 judge[real_mask]=[T, F, T] judge=[T,F,F,T,T] real_mask=[T,F,T,T,F]
        acc_fake = np.count_nonzero(judge[fake_mask]) / np.count_nonzero(fake_mask)  # 在 NumPy / Python 中，布尔值 True 可以视为 1，False 可以视为 0。
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader): # 这个函数只负责一个测试集
        test_recorder_loss = defaultdict(Recorder)  # define test recorderc
        prediction_lists, feature_lists, label_lists = [], [], []  # 分别保存预测概率、特征、标签的列表
        for i, batch_data in tqdm(enumerate(data_loader),total=len(data_loader)):  # 遍历测试集中的 batch；测试阶段不执行 optimizer.step()
            if 'label_spe' in batch_data: batch_data.pop('label_spe')  # get data 删除特定类别标签 字典删除键值对的语法
            batch_data['label'] = torch.where(batch_data['label']!=0, 1, 0)  # 把所有非零标签统一变成 fake 这说明当前 Trainer 的评测指标主要面向二分类检测。
            for key, value in batch_data.items():
                if value is not None: batch_data[key] = value.cuda()
            predictions = self.inference(batch_data) # 执行无梯度推理  因此这里不会建立反向传播计算图。第一个 batch：prediction_lists = [0.1, 0.8, 0.4, 0.9] 第二个batch：prediction_lists = [0.1, 0.8, 0.4, 0.9,0.2, 0.7, 0.6, 0.3]
            label_lists.extend(list(batch_data['label'].cpu().detach().numpy()))  # GPU Tensor → CPU Tensor → NumPy Array → Python List
            prediction_lists.extend(list(predictions['prob'].cpu().detach().numpy())) # 每个 batch 预测完以后，把结果累计保存起来。
            feature_lists.extend(list(predictions['feat'].cpu().detach().numpy())) if self.config['save_feat'] else None  # 配置允许时才累计当前batch的特征，否则不修改feature_lists
            if type(self.model) is not AveragedModel:# 测试阶段仍然计算 loss 测试loss不参与训练，只是作为是否过拟合的一个参考
                losses = self.model.module.get_losses(batch_data, predictions) if type(self.model) is DDP else self.model.get_losses(batch_data, predictions)
                for name, value in losses.items(): test_recorder_loss[name].update(value)  
        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists) # shape=(10000,)→一维向量，长度10000  并非shape=(10000, 1)→二维数组，10000行 × 1列

    def save_best(
            self,  # 当前Trainer实例，持有模型、配置、历史最佳指标和日志目录
            epoch,  # 当前训练轮次，用于标记最佳模型产生在哪个epoch
            iteration,  # 当前epoch内的batch序号，与epoch共同定位最佳模型产生时刻
            step,  # 从训练开始累计的全局batch步数，作为TensorBoard横坐标
            losses_one_dataset_recorder,  # 当前测试集的各项loss记录器；计算avg时传入None
            key,  # 当前测试集名称，也是最佳指标字典和保存目录的索引；'avg'表示多测试集平均结果
            metric_one_dataset,  # 单个测试集时含5项标量指标及逐样本pred/label；key='avg'时含跨测试集平均指标及dataset_dict
            features_one_dataset  # 当前测试集汇总后的特征数组；'avg'没有对应特征，因此传入None
            ):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring, float('-inf') if self.metric_scoring != 'eer' else float('inf'))
        improved = metric_one_dataset[self.metric_scoring] < best_metric if self.metric_scoring == 'eer' else metric_one_dataset[self.metric_scoring] > best_metric
        if improved:    # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring] #值更新
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict'] #dataset_dict 字典更新
            if self.config['save_ckpt'] and key not in FFpp_pool: # 配置中权重和特征需要保存才保存，指标字典默认保存
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            if self.config['save_feat'] and key != 'avg':
                self.save_feat('test', features_one_dataset, key)  # 只保存当前测试集最佳指标对应的特征，avg不对应单一特征矩阵
            self.save_metrics('test', metric_one_dataset, key)

        metric_str = f"dataset:{key} step:{step}"  # avg没有loss记录器，但仍需要初始化指标日志
        if losses_one_dataset_recorder is not None: #记录所有测试指标
            loss_str = f"dataset:{key} step:{step}"
            for k, v in losses_one_dataset_recorder.items(): # K = overall,cls_loss,contrastive_loss etc.
                writer = self.get_writer('test', key, k) #  阶段 + 数据集 + 指标 对应的 TensorBoard 写入器 
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)# tensorboard-1. loss 
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str) # 例 dataset: Celeb-DF-v2 step: 1800 testing-loss, overall: 0.43
            for k, v in metric_one_dataset.items():
                if k == 'pred' or k == 'label' or k == 'dataset_dict': # 因为这些 pred label dataset_dict不是单个标量。
                    continue
                metric_str += f"testing-metric, {k}: {v}"
                writer = self.get_writer('test', key, k)# tensorboard-2. metric
                writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset: # 单个测试集：含 pred/label，可以计算分类别准确率。avg 虚拟测试集：只有五项平均指标，没有逐样本 pred/label，因此跳过。
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer = self.get_writer('test', key, 'class_acc')  # 为两类准确率单独创建Writer，避免复用前面metric循环遗留的writer
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)


    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()  # 将模型设置为评估模式（关闭Dropout、BatchNorm的train模式行为）
        losses_all_datasets = {}   # 初始化结果记录容器 存储所有测试集的损失值（字典格式：数据集名->损失列表）
        metrics_all_datasets = {}  # 存储所有测试集的评估指标（字典格式：数据集名->指标字典）
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}

        keys = test_data_loaders.keys()  # 获取所有测试数据集的名称列表
        for key in keys:  # 逐个处理每个测试集 每次 key 是一个测试集名称
            dataset_metadata = test_data_loaders[key].dataset.data_dict
            self.save_dataset_metadata('test', dataset_metadata, key)
            losses_one_dataset_recorder, predictions_nps, label_nps, features_nps = self.test_one_dataset(test_data_loaders[key]) # 在当前测试集上进行推理并获取结果 test_one_dataset返回：损失记录器、预测结果numpy数组、标签numpy数组、特征numpy数组
            losses_all_datasets[key] = losses_one_dataset_recorder  # 记录当前数据集的损失值
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=dataset_metadata['image'])# 计算当前测试集的各项评估指标（传入预测结果、真实标签和图像名称）
            for metric_name, value in metric_one_dataset.items():# 累加指标到平均统计（只处理预定义的指标）
                if metric_name in avg_metric: avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]   #记录当前数据集的主评估指标到dataset_dict（用于后续模型选择） self.metric_scoping指定主指标（如AUC）
            if type(self.model) is AveragedModel:  # 判断是否为SWA模型 特殊处理SWA模型（随机权重平均模型只需记录指标，不参与最佳模型保存）
                metric_str = f"Iter Final for SWA:    "  # SWA模型的最终迭代标识
                for metric_name, value in metric_one_dataset.items():
                    metric_str += f"testing-metric, {metric_name}: {value}"  # 拼接所有指标值
                self.logger.info(metric_str)  # 输出日志
                continue  # 跳过后续模型保存步骤
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset,features_nps) # 每个测试集独立指标及对应特征最佳保存（如果当前指标优于历史最佳）
       
        if len(keys)>0 and self.config.get('save_avg',False):  #当配置开启保存平均指标时（需所有数据集处理完成后执行） 确保至少有一个数据集且配置允许
            for key in avg_metric:
                if key != 'dataset_dict':avg_metric[key] /= len(keys)  # 算各指标的平均值（总累计值/数据集数量)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric, None)# 将平均指标视为一个虚拟数据集进行保存（avg没有对应的单一特征矩阵）

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, batch_data):
        predictions = self.model(batch_data, inference=True)
        return predictions
