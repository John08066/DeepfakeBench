import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # DeepfakeBench 项目根目录
TRAINING_ROOT = PROJECT_ROOT / "training"  # 训练代码与配置目录

DEFAULT_DATA_ROOT = Path("/home/zhaoting.ding/disk/Datasets")  # 实验室共享数据集根目录
DEFAULT_PRETRAINED_ROOT = PROJECT_ROOT.parent / "pretrained"  # PRD 预训练权重根目录


def _root_from_config(config, config_key, env_key, default):
    value = os.environ.get(env_key) or config.get(config_key) or default
    return Path(value).expanduser().resolve()


def resolve_data_paths(config):
    data_root = _root_from_config(
        config, "data_root", "DEEPFAKE_DATA_ROOT", DEFAULT_DATA_ROOT
    )
    config["data_root"] = str(data_root)
    for key in ("rgb_dir", "lmdb_dir", "dataset_json_folder"):
        path = Path(config[key]).expanduser()
        if not path.is_absolute():
            path = data_root / path
        config[key] = str(path.resolve())
    return config


def resolve_pretrained_path(config, key, default_name):
    pretrained_root = _root_from_config(
        config,
        "pretrained_root",
        "PRD_PRETRAINED_ROOT",
        DEFAULT_PRETRAINED_ROOT,
    )
    path = Path(config.get(key) or default_name).expanduser()
    if not path.is_absolute():
        path = pretrained_root / path
    config["pretrained_root"] = str(pretrained_root)
    config[key] = str(path.resolve())
    return config[key]
