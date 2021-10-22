import random
import numpy as np


def generate(total=200, x_scale=1000, y_scale=1000):
    """
    :param total: 生成的所有点点数, 默认200
    :param x_scale: x轴上的最大取值, 最小为0
    :param y_scale: x轴上的最大取值, 最小为0
    :return: 返回x轴坐标点集合和y轴坐标点数组, 例如有两个点(1, 2), (3, 4), 返回的是[1 3],[2 4]
    """
    x = np.random.randint(x_scale, size=total)
    y = np.random.randint(y_scale, size=total)
    return x, y
