# import pickle
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.manifold import TSNE
#
# # 定义初始颜色映射
# color_map = {}
#
#
# def generate_colors(unique_label_spe):
#     # 生成更多的颜色
#     cmap = plt.get_cmap('Set2')  # 使用 tab20 调色板
#     colors = [cmap(i) for i in range(len(unique_label_spe))]
#     return {label: color for label, color in zip(unique_label_spe, colors)}
#
#
# def tsne_draw(x_transformed, label, spe_label, ax, epoch=0, log='', pred_label=None):
#     tsne_df = pd.DataFrame(x_transformed, columns=['X', 'Y'])
#     tsne_df["Label"] = label
#     tsne_df["Spe_Label"] = spe_label
#
#     marker_list = ['*' if l == 0 else 'o' for l in tsne_df["Label"]]
#
#     if pred_label is not None:
#         tsne_df["Pred_Label"] = pred_label
#         tsne_df["Correct"] = tsne_df["Label"] == (tsne_df["Pred_Label"] > 0.5)
#         correct_colors = ['green' if c else 'red' for c in tsne_df["Correct"]]
#         for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], correct_colors, marker_list):
#             ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m)
#     else:
#         for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], [color_map[i] for i in tsne_df["Spe_Label"]], marker_list):
#             ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m)
#
#     print(f'epoch{epoch} ' + log)
#     ax.axis('off')
#
#
# # 只处理一个 tsne_dict
# tsne_dict_path = '/home/csy/disk1/project_1/deepfakeBench/DeepfakeBench-main/training/tsne/our_x2.pkl'
# dataset_name = 'FaceForensics++'
# name = f'RWS_{dataset_name}'
#
# print(f'Processing {tsne_dict_path}...')
# np.random.seed(0)
# tsne = TSNE(n_components=2, random_state=1024)
#
# with open(tsne_dict_path, 'rb') as f:
#     tsne_dict = pickle.load(f)
#
# feat = tsne_dict[dataset_name]['feat'].reshape((tsne_dict[dataset_name]['feat'].shape[0], -1))
# label_spe = np.array(tsne_dict[dataset_name]['spe_label'])
# label = np.array(tsne_dict[dataset_name]['label'])
# pred_label = np.array(tsne_dict[dataset_name]['pred'])
#
# # 筛选出 FF-real, FF-DF, FF-NT 的数据
# selected_labels = np.unique(label_spe)
# selected_indices = np.isin(label_spe, selected_labels)
# feat = feat[selected_indices]
# label_spe = label_spe[selected_indices]
# label = label[selected_indices]
# pred_label = pred_label[selected_indices]
#
# label_0_indices = np.where(label == 0)[0][:2500]
# other_label_indices = np.where(label != 0)[0]
# num_samples = len(label_0_indices)
# other_label_indices_sampled = np.random.choice(other_label_indices, size=num_samples, replace=False)
# sampled_indices = np.concatenate((label_0_indices, other_label_indices_sampled))
# np.random.shuffle(sampled_indices)
#
# feat = feat[sampled_indices]
# label_spe = label_spe[sampled_indices]
# label = label[sampled_indices]
# pred_label = pred_label[sampled_indices]
#
# feat_transformed = tsne.fit_transform(feat)
#
#
# # 创建两个子图
# fig, axs = plt.subplots(1, 2, figsize=(20, 10))
#
# # 第一个子图：根据真实标签绘制
# # axs[0].set_title(f'T-SNE of {name} (Ground truth Labels)')
#
# # 创建图例句柄和标签
# unique_label_spe = sorted(set(label_spe))
#
# # 生成颜色映射
# color_map = generate_colors(unique_label_spe)
#
# tsne_draw(feat_transformed, label, label_spe, ax=axs[0], epoch=0, log='share_in_specific')
#
# # 创建图例句柄和标签
# # handles = [
# #     plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color_map[i], markersize=10) if i == 0 else plt.Line2D(
# #         [0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=10) for i in unique_label_spe]
# # labels = unique_label_spe
# #
# # fig.legend(handles, labels, title="Classes", loc="upper left", fontsize=14)
#
# # 第二个子图：根据预测结果绘制
# # axs[1].set_title(f'T-SNE of {name} (Predicted Labels)')
#
# tsne_draw(feat_transformed, label, label_spe, ax=axs[1], epoch=0, log='share_in_specific', pred_label=pred_label)
#
# # 创建图例句柄和标签
# # handles_pred = [
# #     plt.Line2D([0], [0], marker='*', color='green', markersize=10),
# #     plt.Line2D([0], [0], marker='*', color='red', markersize=10),
# #     plt.Line2D([0], [0], marker='o', color='green', markersize=10),
# #     plt.Line2D([0], [0], marker='o', color='red', markersize=10),
# # ]
# # labels_pred = ['Correct Real', 'Incorrect Real', 'Correct Fake', 'Incorrect Fake']
#
# # fig.legend(handles_pred, labels_pred, title="Prediction", loc="upper right", fontsize=14)
#
# plt.tight_layout()
# plt.show()
# # plt.savefig(f'TSNE_of_{name}.png')

































import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

# --- (来自你脚本的辅助函数) ---

# 定义全局颜色映射
color_map = {}


def generate_colors(unique_label_spe):
    """根据传入的唯一标签列表，生成一个颜色映射字典。"""
    # 使用 'Set2' 调色板，如果标签多，可以换成 'tab20'
    cmap = plt.get_cmap('Set2')
    colors = [cmap(i) for i in range(len(unique_label_spe))]
    return {label: color for label, color in zip(unique_label_spe, colors)}


def tsne_draw(x_transformed, label, spe_label, ax):
    """
    在指定的 matplotlib 轴 (ax) 上绘制 t-SNE 散点图。
    (注意：已移除 pred_label，因为我们只绘制真实标签)
    """
    tsne_df = pd.DataFrame(x_transformed, columns=['X', 'Y'])
    tsne_df["Label"] = label
    tsne_df["Spe_Label"] = spe_label

    # 根据二分类标签 'Label' (0=real, 1=fake) 生成标记列表
    # 真实样本 (label=0) 使用星号 '*', 伪造样本 (label!=0) 使用圆形 'o'
    marker_list = ['*' if l == 0 else 'o' for l in tsne_df["Label"]]

    # --- 逻辑分支 1: 按种类标签绘图 ---
    # 遍历所有点，使用 'spe_label' 对应的颜色 (从全局 color_map 获取) 和“真/假”的标记来绘制
    for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], [color_map[i] for i in tsne_df["Spe_Label"]], marker_list):
        ax.scatter(_x, _y, color=_c, s=35, alpha=0.7, marker=_m)

    ax.axis('off')


# --- (新功能：数据处理辅助函数) ---

def load_and_process_data(pkl_path, dataset_name, tsne_model, num_samples_per_class=2000):
    """
    加载 .pkl 文件, 提取数据, 采样, 并运行 t-SNE。
    """
    print(f"Processing {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        tsne_dict = pickle.load(f)

    # --- 1. 数据提取 ---
    feat = tsne_dict[dataset_name]['feat'].reshape((tsne_dict[dataset_name]['feat'].shape[0], -1))
    label_spe = np.array(tsne_dict[dataset_name]['spe_label'])
    label = np.array(tsne_dict[dataset_name]['label'])

    # --- 2. 数据采样 (平衡 real 和 fake) ---
    label_0_indices = np.where(label == 0)[0]
    # 如果 real 样本多于 num_samples_per_class，则从中抽样，否则全要
    if len(label_0_indices) > num_samples_per_class:
        label_0_indices = np.random.choice(label_0_indices, size=num_samples_per_class, replace=False)

    other_label_indices = np.where(label != 0)[0]
    # 从 fake 样本中抽样
    num_fake_samples = min(len(other_label_indices), num_samples_per_class)
    other_label_indices_sampled = np.random.choice(other_label_indices, size=num_fake_samples, replace=False)

    # 合并索引并打乱
    sampled_indices = np.concatenate((label_0_indices, other_label_indices_sampled))
    np.random.shuffle(sampled_indices)

    # --- 3. 应用采样索引 ---
    feat = feat[sampled_indices]
    label_spe = label_spe[sampled_indices]
    label = label[sampled_indices]

    # --- 4. 执行 t-SNE 降维 ---
    print("Running t-SNE... (This may take a moment)")
    feat_transformed = tsne_model.fit_transform(feat)

    return feat_transformed, label, label_spe


# --- 主程序开始 ---

# --- 1. 定义路径和参数 ---
baseline_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/tsne/lora.pkl'
our_path = '/root/csy-7pw03c/disk/project/DeepfakeBench-main/training/tsne/our_orig.pkl'
dataset_name = 'FaceForensics++'  # 假设两个 pkl 都用这个 key
save_name = f'TSNE_Comparison_{dataset_name}.png'

# ‼️ 重要：请根据你的 'spe_label' 标签 (0, 1, 2...) 检查这个映射
# 这将用于底部的图例
label_name_map = {
    0: 'Real',  # 对应 FF-real
    1: 'F2F',  # 对应 FF-F2F
    2: 'DF',   # 对应 FF-DF
    3: 'FS',   # 对应 FF-FS
    4: 'NT',   # 对应 FF-NT
    # 5: 'Aug' # 如果你有第6类 (比如 'Aug')，取消注释
}

# --- 2. 初始化 ---
np.random.seed(1023)  # 保证采样和t-SNE可复现
tsne = TSNE(n_components=2, random_state=1023, perplexity=30, n_iter=1000)
fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # 1行2列的画布

# --- 3. 处理并绘制 Baseline (左图) ---
base_trans, base_label, base_spe = load_and_process_data(
    baseline_path, dataset_name, tsne, num_samples_per_class=2500
)

# --- 4. 处理并绘制 Ours (右图) ---
our_trans, our_label, our_spe = load_and_process_data(
    our_path, dataset_name, tsne, num_samples_per_class=2500
)

# --- 5. 生成全局颜色映射 ---
# (使用两个数据集中所有出现过的标签，来保证颜色统一)
all_unique_labels = sorted(list(set(base_spe) | set(our_spe)))
color_map = generate_colors(all_unique_labels)  # 设置全局 color_map

# --- 6. 绘制 ---
print("Drawing plots...")
# 绘制左图
axs[0].set_title('Clip', fontsize=24, pad=20)
tsne_draw(base_trans, base_label, base_spe, ax=axs[0])

# 绘制右图
axs[1].set_title('Ours', fontsize=24, pad=20)
tsne_draw(our_trans, our_label, our_spe, ax=axs[1])

# --- 7. 创建共享图例 (像示例图片一样) ---
handles = []
labels = []
for spe_label in all_unique_labels:
    # 假设 0 是 'Real'，使用星号
    marker = '*' if spe_label == 0 else 'o'
    # 从全局 color_map 获取颜色
    color = color_map[spe_label]
    # 从你定义的 name_map 获取名字
    name = label_name_map.get(spe_label, f'Label {spe_label}')

    # 创建一个图例条目
    handles.append(plt.Line2D([0], [0], marker=marker, color='w',
                              markerfacecolor=color, markersize=15))
    labels.append(name)

# 将图例放在画布底部中央
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.02),  # 调整位置
           ncol=len(all_unique_labels),  # 一行显示
           fontsize=20, markerscale=1.5)

# --- 8. 保存图像 ---
# 调整布局，为底部的图例留出空间
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
plt.savefig(save_name, bbox_inches='tight')

# print(f"Comparison plot saved to {save_name}")