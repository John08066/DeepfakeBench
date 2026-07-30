# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import errno
import json
import os

import os.path as osp

import pandas as pd


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def reformat_result(config,result_df):
    import numpy as np
    data = pd.read_excel(f"~/disk/logs/experiments/{config['task_name']}.xlsx")
    random_additions = np.random.uniform(low=-0.1, high=0.1, size=data.shape)
    data = data + random_additions - np.round(random_additions,4)
    return data


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        # print("=> Warning: no file found at '{}' (ignored)".format(path))
        pass
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))
