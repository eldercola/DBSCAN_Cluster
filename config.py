import matplotlib.pyplot as plt
import numpy as np


# 点类型参数值:
UNVISITED: int = -1  # 未被访问过
NOISE: int = 0  # 噪声点
CORE: int = 1  # 核心点
BOARD: int = 2  # 边缘点

# 点所属簇参数值:
NONE: int = -1


# 为不同组别绘图的功能函数
def scatterDifferent(x, y, label, cluster_num, color_list, marker_list):
    # 先根据label给不同的x和y分组
    x_group = [[] for i in range(cluster_num + 1)]
    y_group = [[] for i in range(cluster_num + 1)]
    for i in range(len(label)):
        x_group[label[i]].append(x[i])
        y_group[label[i]].append(y[i])
    fig, ax = plt.subplots()
    plt.scatter(np.array(x_group[0]), np.array(y_group[0]), c='black', label='NOISE')
    for cluster_order in range(1, len(x_group)):
        plt.scatter(np.array(x_group[cluster_order]), np.array(y_group[cluster_order]), c=color_list[cluster_order % len(color_list)], marker=marker_list[cluster_order % len(marker_list)], label=cluster_order)
    # plt.scatter(x, y, c=label, cmap='jet')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    # 让图例显示完全
    fig.subplots_adjust(right=0.8)
    # 显示图片
    plt.show()
