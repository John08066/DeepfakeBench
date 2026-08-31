import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------- 数据 ----------
data = {
    "Change Saturation": {
        "ours": [0.9833, 0.9833, 0.9828, 0.9832, 0.9830],
        "spsl": [0.9726, 0.9708, 0.9686, 0.9669, 0.9666],
        "sbi":  [0.8991, 0.8971, 0.8984, 0.8960, 0.8929],
        "lsda": [0.9446, 0.9403, 0.9362, 0.9315, 0.9264],
    },
    "Change Contrast": {
        "ours": [0.9793, 0.9737, 0.9676, 0.9567, 0.9406],
        "spsl": [0.9714, 0.9631, 0.9478, 0.9232, 0.8892],
        "sbi":  [0.8810, 0.8711, 0.8512, 0.8099, 0.7583],
        "lsda": [0.9394, 0.9285, 0.9156, 0.8958, 0.8573],
    },
    "Block Wise": {
        "ours": [0.9882, 0.9844, 0.9810, 0.9731, 0.9676],
        "spsl": [0.9738, 0.9688, 0.9636, 0.9625, 0.9520],
        "sbi":  [0.8967, 0.8991, 0.9066, 0.9049, 0.9001],
        "lsda": [0.9548, 0.9617, 0.9569, 0.9559, 0.9490],
    },
    "Gaussian Blur": {
        "ours": [0.9644, 0.9396, 0.8661, 0.7727, 0.6658],
        "spsl": [0.9783, 0.9628, 0.8949, 0.7473, 0.6143],
        "sbi":  [0.7553, 0.6647, 0.5311, 0.4910, 0.4547],
        "lsda": [0.8649, 0.7993, 0.7033, 0.6386, 0.5994],
    },
    "JPEG Compression": {
        "ours": [0.9637, 0.9476, 0.9144, 0.8664, 0.8196],
        "spsl": [0.9725, 0.9496, 0.9128, 0.8629, 0.8012],
        "sbi":  [0.8459, 0.8073, 0.8196, 0.6419, 0.6214],
        "lsda": [0.8891, 0.8611, 0.8386, 0.8013, 0.7586],
    },
}

# 计算 Average
distortions = ["Gaussian Blur", "Block Wise", "Change Contrast",
               "Change Saturation", "JPEG Compression"]
methods = ["lsda", "spsl", "sbi", "ours"]
avg = {m: np.array([data[d][m] for d in distortions]).mean(axis=0).tolist() for m in methods}
data["Average"] = avg

legend_labels = {"lsda": "LSDA", "spsl": "SPSL", "sbi": "SBI", "ours": "Ours"}
styles = {
    "lsda": dict(color="tab:red",    marker="o"),
    "spsl": dict(color="tab:orange", marker="x"),
    "sbi":  dict(color="tab:green",  marker="s"),
    "ours": dict(color="tab:blue",   marker="^"),
}

# 面板顺序（把 Change Contrast 与 Gaussian Blur 对调）
panel_order = ["Change Contrast", "Block Wise", "Gaussian Blur",
               "Change Saturation", "JPEG Compression", "Average"]

# ---------- 绘图 ----------
x = np.arange(5)  # 0..4 -> Level 1..5
fig, axes = plt.subplots(1, 6, figsize=(18, 4.2), sharex=True, sharey=True)

for ax, name in zip(axes, panel_order):
    ax.set_title(name, fontsize=10)

    # 让纵轴（spine）相对横轴更长：提高图框的高宽比 (>1)
    # 需要 Matplotlib 3.3+
    ax.set_box_aspect(1.18)  # 高/宽 = 1.18，纵轴可视上更“长”

    # 折线样式（不加粗，统一）
    for m in methods:
        y = np.array(data[name][m]) * 100.0
        ax.plot(x, y, linewidth=1.0, markersize=5,
                color=styles[m]["color"], marker=styles[m]["marker"])

    # 轴范围/刻度/标签
    ax.set_xlim(-0.1, 4.1)                 # 左右留细缝
    ax.set_xticks([0, 2, 4])
    ax.set_xlabel("Level")
    ax.set_ylim(45, 100)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_ylabel("AUC (%)")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)

    # 刻度线长度：横轴更短，纵轴略长（但不加粗）
    ax.tick_params(axis='x', length=2, width=0.8, direction='out', labelbottom=True)
    ax.tick_params(axis='y', length=4, width=0.8, direction='out', labelleft=True)

    # 轴线统一细线
    for s in ax.spines.values():
        s.set_linewidth(0.8)

    # （可选）若想让纵轴视觉上“超出”上下边界一点点，取消下面两行注释：
    # ax.spines['left'].set_visible(False)
    # ax.plot([0, 0], [-0.03, 1.03], transform=ax.transAxes, clip_on=False, color='black', linewidth=0.8)

# 底部共享图例
handles = [Line2D([0], [0], color=styles[m]["color"], marker=styles[m]["marker"],
                  linewidth=1.0, markersize=5, label=legend_labels[m]) for m in methods]
fig.legend(handles=handles, labels=[legend_labels[m] for m in methods],
           ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("robustness_panels_refstyle_v5.png", dpi=300, bbox_inches="tight")
plt.show()
