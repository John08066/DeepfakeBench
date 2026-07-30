import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_difference(pkl_path, save_path='diff_distribution_mae.png'):
    print(f"正在从 {pkl_path} 加载特征数据...")

    if not os.path.exists(pkl_path):
        print(f"错误: 找不到文件 {pkl_path}，请检查路径。")
        return

    # 1. 读取 Pickle 文件
    with open(pkl_path, 'rb') as f:
        tsne_dict = pickle.load(f)

    all_feats = []
    all_labels = []

    # 2. 遍历字典，将所有数据集的特征和标签拼接起来
    for dataset_key, data in tsne_dict.items():
        all_feats.append(data['feat'])
        all_labels.append(data['label'])

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 3. 核心计算：对特征取绝对值，然后求均值 (Mean Absolute Error)
    # feats 的形状是 (N, C)，例如 (1000, 1024)
    # 结果 diff_magnitude 是一个形状为 (N,) 的一维数组，代表每张图的差异大小
    diff_magnitude = np.abs(feats).mean(axis=1)

    # 4. 根据标签分离真假数据
    # DeepfakeBench 的标准习惯通常是 0 代表 Real (真图)，1 代表 Fake (假图)
    # 如果你画出来的图反了，把这里的 0 和 1 对调一下即可
    real_diffs = diff_magnitude[labels == 0]
    fake_diffs = diff_magnitude[labels == 1]

    print("-" * 30)
    print(f"真实图像 (Real) 数量: {len(real_diffs)}, 差异均值: {real_diffs.mean():.4f}")
    print(f"伪造图像 (Fake) 数量: {len(fake_diffs)}, 差异均值: {fake_diffs.mean():.4f}")
    print("-" * 30)

    # 5. 开始绘图
    plt.figure(figsize=(10, 6))

    # 使用 seaborn 绘制核密度估计图 (KDE)
    # fake 的差异小（偏左），用红色；real 的差异大（偏右），用蓝色
    sns.kdeplot(fake_diffs, color='red', fill=True, label='Fake', alpha=0.5, linewidth=2)
    sns.kdeplot(real_diffs, color='blue', fill=True, label='Real', alpha=0.5, linewidth=2)

    # 设置图表标题和标签
    plt.title('Distribution of VAE Reconstruction Feature Differences (MAE)', fontsize=16)
    plt.xlabel('Feature Difference Magnitude (Absolute Mean)', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    # 添加图例和网格
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 限制 X 轴最小值为 0 (因为绝对值均值不可能为负)
    plt.xlim(left=0)

    # 紧凑布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"图表已成功保存至: {save_path}")

    # 如果是在带图形界面的环境下，可以取消注释下面这行来显示图表
    # plt.show()


if __name__ == '__main__':
    # 这里直接使用了你测试代码里的 pkl 保存路径
    pkl_file = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/tsne/our_diff.pkl'

    # 运行画图函数，输出图片将保存在当前目录下的 diff_distribution_mae.png
    plot_feature_difference(pkl_file)