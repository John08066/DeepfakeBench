import os
import logging

import torch.distributed as dist

class RankFilter(logging.Filter): #只有当前进程 rank 等于实例传入的 rank，日志才输出，其他进程全部吞掉日志。
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return dist.get_rank() == self.rank


def create_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)# Create log path (exist_ok=True already handles the case where it exists)

    logger = logging.getLogger()    # 没有传名字，拿到的是 Python logging 的 root logger（根日志器） 整个项目的“总日志入口”
    logger.setLevel(logging.INFO)   #设置这个 logger 对象的日志过滤等级，决定哪些级别的日志会被放行往下传给 handler，哪些直接丢弃。只允许级别 ≥20（INFO 及以上）的日志通过 logger，DEBUG(10)会直接被过滤掉，不会交给后面的
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')# 格式类似 2026-08-18 15:42:10,345 - INFO - start training model

    fh = logging.FileHandler(log_path) #创建一个文件出口：日志 → fh → training.log
    fh.setFormatter(formatter) # 多个 handler 可以复用同一个 formatter 对象，就像 fh 和 sh 共用
    logger.addHandler(fh)  # 给总日志入口装一个“往文件里写”的出口。

    sh = logging.StreamHandler() #创建一个文件出口：日志 → fh → 终端
    sh.setLevel(logging.INFO)  # 两层级别控制 算 logger 放行，每个 handler（文件输出 fh、控制台输出 sh）还可以再设置一遍过滤等级。
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger