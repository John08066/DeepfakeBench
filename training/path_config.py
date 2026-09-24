import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # DeepfakeBench 项目根目录
TRAINING_ROOT = PROJECT_ROOT / "training"  # 训练代码与配置目录

DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_PRETRAINED_ROOT = PROJECT_ROOT / "pretrained"


def project_path(value):
    """Resolve repository-relative paths independently of the shell directory."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _root_from_config(config, config_key, env_key, default):
    value = os.environ.get(env_key) or config.get(config_key) or default
    return project_path(value)


def resolve_output_path(value):
    """Output paths are relative to PRD_OUTPUT_ROOT (the repository by default)."""
    root = project_path(os.environ.get("PRD_OUTPUT_ROOT") or PROJECT_ROOT)
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def resolve_rgb_path(value, rgb_root):
    """JSON frame paths are relative to the configured RGB root, not cwd."""
    value = value.replace('\\', '/')
    if value.startswith('./datasets/'):
        value = value[len('./datasets/'):]
    path = Path(value)
    return str(path if path.is_absolute() else Path(rgb_root) / path)


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
