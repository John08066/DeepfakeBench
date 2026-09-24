# PRD 通用开发基线

本轮移植范围是 PRD 正式训练、测试、路径解析与启动检查，不是对 DeepfakeBench 全部历史 detector、预处理和可视化工具的重新验证。数据和权重不进入 Git；各机器只配置存储根目录，实验协议保持不变。

## 1. 共同实验协议

默认使用 `training/config/detector/prd_probe_ablation.yaml`，即已有 T1 机制实验基线：

- FF++ c23 训练；Celeb-DF-v2、DFDCP、DFDC 测试。
- CLIP ViT-L/14 + LoRA、原始 VAE 探针、有符号残差、原图与残差特征拼接。
- train batch 32 / test batch 64；seed 1024；每视频训练 8 帧、测试 32 帧；分辨率 224。
- `early_stopping.enabled: true`、`max_declines: 3`。这是 epoch 末指标连续三次比上一轮恶化，不是三轮未刷新历史最优；持平或回升会清空计数。
- 保留既有 `start_epoch: 0`、`nEpochs: 30` 和包含终点的训练循环语义，即未早停时 epoch 0–30，共 31 轮；不在移植时悄悄变成 30 轮。

`lora.yaml` 保留原始复现对照：train 32 / test 16，早停显式关闭。请不要混称两个配置为同一个协议。探针、残差及特征消融配置是实验方向，不自动替代共同基线。固定 seed 不代表跨 GPU、驱动和 cuDNN 版本逐位一致。

沿用历史协议，三个测试集的平均 AUC 参与早停和最佳 checkpoint 选择；这里的平均是同一 checkpoint 在三个数据集的指标平均，不是权重平均。这三个集因此不是独立的最终盲测集。论文结论需要另行固定验证/最终测试协议，不能把工程迁移成功当成实验有效性证明。

## 2. 每台服务器只配置三类位置

目录默认相对仓库根解析，与启动命令的当前目录无关。也允许在机器环境变量中指定绝对存储位置；禁止的是把某台机器的绝对路径写死进共享代码。

| 环境变量 | 默认位置 | 内容 |
| --- | --- | --- |
| `DEEPFAKE_DATA_ROOT` | 仓库下 `datasets/` | 数据目录与 JSON 索引的共同父目录 |
| `PRD_PRETRAINED_ROOT` | 仓库下 `pretrained/` | 已下载的完整 CLIP、VAE 模型目录 |
| `PRD_OUTPUT_ROOT` | 仓库根 | 配置中相对输出路径的基准目录，不是模型目录本身 |

环境变量优先于 YAML 中的根目录设置。例：数据根下必须保持以下相对结构，而不是只准备原始视频：

```text
datasets/
└── DeepfakeBench/
    ├── lmdb/
    │   ├── FaceForensics++_lmdb/data.mdb
    │   ├── Celeb-DF-v2_lmdb/data.mdb
    │   ├── DFDCP_lmdb/data.mdb
    │   └── DFDC_lmdb/data.mdb
    └── config/dataset_json/
        ├── FaceForensics++.json
        ├── Celeb-DF-v2.json
        ├── DFDCP.json
        └── DFDC.json
pretrained/
├── clip-vit-large-patch14/  # config.json + 完整模型权重
└── vae_original/           # config.json + 完整扩散模型权重
```

LMDB 内部 key 必须与 JSON 中的帧路径匹配。只把目录改成这个名称，并不能把任意原始视频变成可训练数据；这里沿用已经预处理好的数据格式。默认 `train_config.yaml` 使用 LMDB；独立 `test_config.yaml` 保留旧 RGB 默认，测试已有 LMDB 时须显式加 `--lmdb`。

在各机器的终端设置一次（下列示例均相对仓库根，实际可指向挂载盘）：

```bash
export DEEPFAKE_DATA_ROOT=./datasets
export PRD_PRETRAINED_ROOT=./pretrained
export PRD_OUTPUT_ROOT=./outputs/server_a
export CUDA_VISIBLE_DEVICES=0
```

这些机器选择不要写回共享 YAML。没有设置环境变量时，准备好默认目录即可。数据、权重或 GPU 不满足要求时应先修正检查失败，不临时改实验协议来掩盖问题。

## 3. 环境安装证据与边界

2026-09-24 只读核对了 4090 现有 PRD 环境：Python 3.10.21、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124；`python -m pip check` 返回 `No broken requirements found.`。`requirements-prd.txt` 的版本来自该环境的 `importlib.metadata`，不是猜测或自动升级到最新版。

2026-09-25 在 203-1 的隔离 `prd-common` 环境（Python 3.10.21）已完成依赖安装，`pip check`、39 项 CPU 回归测试及无卡导入通过。该机器复用了已有的 CUDA 12.4 依赖包，并从 4090 复制了匹配的 dlib wheel；这不是对任意新服务器执行下述在线安装命令的完整验证。203-1 当前无 GPU、数据和预训练权重，尚未运行真实训练 smoke。该文件固定核心/运行入口依赖，但不是系统库、驱动及所有间接包的完整锁文件。当前 detector/dataset 注册模块会预先导入其他算法，因此包含 dlib、timm、fvcore 等兼容依赖；不要只安装 torch 和 transformers 就认为入口齐全。旧 `install.sh` / `Dockerfile` 对应上游历史环境，不等价于本轮 PRD 环境。

在新的、独立的 Python 3.10 环境中安装，不覆盖另一实验的环境：

```bash
conda create -n prd-common python=3.10.21 -y
conda activate prd-common
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-prd.txt
python -m pip check
```

PyTorch 的版本组合与 CUDA 12.4 wheel 源由[官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v251)核对。新服务器的 NVIDIA 驱动必须支持选定 wheel，显存也必须满足 batch 32/64；不能只看本机装了某个 CUDA toolkit 就认定兼容。dlib 若无适配 wheel，安装过程可能需要 CMake/C++ 编译器；OpenCV 可能需要系统共享库。遇到这些问题应保留完整安装错误单独解决，不自动替换算法或升级 NumPy。

安装需要网络，但正式训练已准备好的本地模型无需在线下载。离线机器应事先准备相同版本的安装包、完整权重和数据。

## 4. 先检查，再训练

从仓库根运行：

```bash
python scripts/check_prd_environment.py \
  --config training/config/detector/prd_probe_ablation.yaml
```

检查通过后，使用一批真实数据验证模型、一次参数更新、测试及 checkpoint 读写：

```bash
python scripts/check_prd_environment.py \
  --config training/config/detector/prd_probe_ablation.yaml --smoke
```

`--smoke` 的 batch 2 是验证资源开销设置，**不等于正式 batch 32/64 的完整训练或显存验收，也不产生可报告的实验指标**。正式运行前确认目标显卡显存、目录容量和机器环境一致。

明确启动一个新实验（使用独一无二的任务名；长训练放在 tmux 会话中）：

```bash
bash scripts/launch_prd_training.sh \
  training/config/detector/prd_probe_ablation.yaml t1_server_a_seed1024
```

不自动启用历史 Codex 监控、cron 或无人值守实验队列。启动脚本的 `PYTHON_BIN` 默认是已激活环境的 `python`；如需指定其他解释器，只在当前机器设置 `PYTHON_BIN`。

独立评测已有 checkpoint 的命令形式：

```bash
python training/test.py \
  --detector_path training/config/detector/prd_probe_ablation.yaml \
  --weights_path YOUR_CHECKPOINT.pth \
  --test_dataset Celeb-DF-v2 DFDCP DFDC --lmdb \
  --output_dir logs/testing/t1_server_a_seed1024 --no-save_tsne
```

`YOUR_CHECKPOINT.pth` 必须替换为该实验真实保存的权重。每次训练应保留 `run_metadata.json`、配置、Git commit、seed、日志和 checkpoint；不要用另一个实验的权重覆盖已有文件。

## 5. 多服务器互不干扰的最小约定

1. `main` 维护审核通过的通用代码；实验按方向命名分支，不按服务器永久分叉。
2. 各服务器从同一明确 commit 建立各自实验分支和独立目录。分支名不同但共享同一个目录，仍不能隔离正在运行的训练。
3. 一旦启动训练，冻结该目录的代码、配置与环境；不在运行目录 `git pull`、切分支或安装新依赖。新开发另用目录/worktree。
4. 每个实验单独任务名和输出根；通用修复合回 main，其他实验在下一次运行前按需吸收，不中途追赶最新 main。
5. 共享只读数据/权重可以节省空间，但不能共享可写日志、checkpoint 或一套随时升级的 Python 环境。

最简单的分工是每台服务器在自己的目录中，从发布后的同一基线 commit 新建一个方向分支：例如 `exp/probe`、`exp/residual`、`exp/encoder`。开始新实验前记录基线 commit；不要把某台服务器正在试验的整条分支直接覆盖另一台。这是多服务器独立单卡实验，不是跨服务器分布式训练。

## 6. 未纳入本轮移植的内容

仓库仍保留其他 detector 配置与实现、Grad-CAM/t-SNE/预测样本可视化、预处理示例以及历史实验摘要。其中还有旧机器绝对路径。历史摘要中的路径是当时实验的证据，不能为了扫描清零而篡改。

2026-09-24 对候选工作树已跟踪文本的机器路径模式扫描，仍有 **280 个命中行 / 86 个文件**，其中 42 行是整行注释。分类是文件所属范围，不表示每一行都会在 PRD 入口运行：

| 范围 | 命中行 | 文件数 |
| --- | ---: | ---: |
| 其他 detector 及其配置 | 97 | 50 |
| 分析、可视化、演示及测试历史注释 | 124 | 12 |
| dataset/library 遗留或示例 | 21 | 10 |
| 预处理 | 4 | 3 |
| 历史实验摘要 | 34 | 11 |

该数字来自 `/home`、`/root`、`/data*`、`/datasets*`、`/mnt`、`/mntcephfs`、`/disk*`、`/Youtu_Pangu*`、`/FaceXray`、`/workspace` 和 Windows 盘符模式，排除 `./datasets` 等相对路径；它不是路径形式的穷尽证明，Linux 系统接口 `/proc` 等也不应一律改成相对路径。

本轮不承诺 `clip_detector.py`、`lora1/lora2`、`lsda`、`dire` 等每个历史算法都可直接运行，也不承诺任意分析或预处理脚本已通用化。需要启用这些方向时，先独立检查它们的配置、权重格式、依赖和默认路径；不能把 PRD 的短程验收扩展解释为整个 DeepfakeBench 全库验收。

## 7. 本次验收记录（2026-09-24）

- 在 4090 新建隔离副本，使用现有 PRD 环境与只读数据/预训练权重；原项目目录、现有 cron 和实验摘要未改动。
- 从仓库外的 `/tmp` 执行，39 项 CPU 回归测试通过：路径迁移、预检、启动器互斥/跨目录隔离、监控状态绑定。
- 开启 Hugging Face 离线模式，真实 LMDB 样本 batch 2 完成前向、反向和一次 Adam 更新；输出形状 `[2, 2]`，约 1.4 GB 完整 checkpoint 保存并严格重新加载成功。
- 另用限样本验收脚本调用原 `train.main()` 和 `test.main()`：训练入口 epoch 0 单步、三个测试集各两个样本、早停记录、元数据、摘要和测试报告均完成。仅验收时缩小数据及批次，正式 YAML 的 batch 32/64、epoch 0–30 不变。
- 没有跑完整基准，没有验证新服务器干净安装、正式 batch 的显存或多卡 DDP；短程指标不可用于论文比较。

基线在独立本地仓库的 `main` 整合，保留 4090 的 `cbc65995a72760bd9a03dc460902a7e14ea6adc5` 与旧 `main` 的 `fdf43927cefe9cd55f03b90605280f473b3fe542` 两条历史。未经单独确认不推送 GitHub；发布前，新服务器直接拉取远端 `main` 还得不到本次改动。
