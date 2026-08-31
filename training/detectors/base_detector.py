# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Abstract Class for the Deepfake Detector

import abc
import torch
import torch.nn as nn
from typing import Union

class AbstractDetector(nn.Module, metaclass=abc.ABCMeta): # metaclass=abc.ABCMeta 这是 Python 的抽象机制 强制子类实现规定函数
    """
    All deepfake detectors should subclass this class.

    Dataset → DataLoader → data_dict{image,label} → Trainer → self.model(data_dict) → AbstractDetector.forward()
    → 具体Detector → pred_dict{prob,feat,logits} → AbstractDetector.get_losses() → loss → Trainer.backward() → optimizer.step()

    """
    def __init__(self, config=None, load_param: Union[bool, str] = False): # Union 是 Python 的类型注解，这个参数允许是多种类型中的任意一种
        """
        config:   (dict)
            configurations for the model
        load_param:  (False | True | Path(str))
            False Do not read; True Read the default path; Path Read the required path
        """
        super().__init__()  #初始化父类 相当于调用：nn.Module.__init__()

    @abc.abstractmethod
    def build_backbone(self, config):
        """
        Builds the backbone of the model.
        """
        pass

    @abc.abstractmethod
    def build_loss(self, config):
        """
        Builds the loss function for the model.
        """
        pass

    @abc.abstractmethod
    def features(self, data_dict: dict) -> torch.tensor:
        """
        Returns the features from the backbone given the input data.
        """
        pass

    @abc.abstractmethod
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """
        Classifies the features into classes.                                                            
        """
        pass

    @abc.abstractmethod
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Returns the losses for the model.所以这个函数必须返回
        {
        'overall': xxx,
        'cls':xxx,
        'rec':xxx
        }
        其中：losses['overall'] 用于backward()
        """
        pass
  
    @abc.abstractmethod
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        Returns the training metrics for the model.
        返回：{
            'acc':0.92,
            'auc':0.95
            }
        """
        pass

    @abc.abstractmethod
    def forward(self, data_dict: dict, inference=False) -> dict: 
        """ 
        为什么输出不是 tensor？因为调用后续get_losses(),get_train_metrics()需要dict类型的dict
        pred_dict={'prob':0.91, 'feat': tensor(...), 'logits': tensor(...)}
        Forward pass through the model, returning the prediction dictionary.
        predictions = self.model(data_dict) 沿着继承链寻找 __call__：于是进入 PyTorch 的 nn.Module.__call__()内部最终会调用 self.forward()
        负责把完整流程串起来  通过调用features()和classifier()来完成前向传播,还有数据预处理，特征归一化等步骤。  这个函数是必须实现的，因为它是模型的核心功能。
        """
        pass
