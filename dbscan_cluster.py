from point import *
import random
import queue
from generate_random_points import *
import matplotlib.pyplot as plt


class dbscan_cluster:
    def __init__(self, x_total, y_total, eps: int, min_pts: int) -> object:
        """
        :param eps: int, 表示半径
        :param min_pts: int, 最少要包含多少点
        """
        self.points = [point(x_total[i], y_total[i]) for i in range(len(x_total))]
        self.wait_for_pick = [i for i in range(len(self.points))]  # 可选取的点
        self.eps = eps
        self.min_pts = min_pts
        self.cur_cluster_num = 0  # 簇序号从0开始

    def printAllPoints(self):
        for p in self.points:
            print('({0}, {1})'.format(p.x_pos, p.y_pos))

    def getClusterNum(self):
        return self.cur_cluster_num

    def initPick(self):
        """
        :return: 在wait_for_pick中随机挑选一个点，返回点在points中的序号
        """
        cur_order = random.randint(0, len(self.wait_for_pick) - 1)  # wait_for_pick中的第cur_order个序号
        cur_pick = self.wait_for_pick[cur_order]  # 存入cur_pick
        self.wait_for_pick.pop(cur_order)  # 再删除
        return cur_pick

    def findNearPoint(self, cur_pick):
        """
        :param cur_pick: int, 表示当前选中的点序号
        :return: 与当前点距离在sqrt(eps^2)内的所有点序号
        """
        near_points = []
        for p in range(len(self.points)):
            if p == cur_pick:  # 同一个点, 跳过
                continue
            else:  # 其他点, 计算一下
                if self.points[cur_pick].calculateDistanceSquare(self.points[p]) <= self.eps ** 2:
                    near_points.append(p)
        return near_points

    def mainProcess(self):
        while len(self.wait_for_pick) > 0:
            # 先随机选取点, 并且在wait_for_pick里面删除其序号
            pre_points = []  # 标记这一个簇已经找到的点
            seeds = queue.Queue()  # 创建一个种子点队列
            cur_pick = self.initPick()  # 获取一个随机点
            near_points = self.findNearPoint(cur_pick)  # 获取随机点的邻近点
            # 当前点的邻近点数量小于 min_pts, 标记为噪声点
            if len(near_points) < self.min_pts:
                self.points[cur_pick].setPointType(NOISE)  # 当前的点是个噪声点
            # 大于 min_pts, 标记为核心点
            elif len(near_points) > self.min_pts:
                self.points[cur_pick].setPointType(CORE)  # 当前的点是核心点
                self.points[cur_pick].setClusterNum(self.cur_cluster_num)  # 设置簇号
                pre_points.append(cur_pick)  # 当前点后续不必再遍历
                for near_point_index in near_points:  # 把它的邻近点全部加入到seeds队列
                    seeds.put(near_point_index)  # 把邻近点在points中的下标加入seeds队列
                    # wait_for_pick队列中需要删除这个邻近点下标
                    if self.wait_for_pick.count(near_point_index) > 0:  # 这个点下标在wait_for_pick队列中
                        self.wait_for_pick.pop(self.wait_for_pick.index(near_point_index))  # 那就pop掉
                while not seeds.empty():  # 只有seeds中还有点数时才继续执行
                    head_point_index = seeds.get()  # 获取队列头部元素, get()方法会在取元素的同时删除队列中的它
                    pre_points.append(head_point_index)  # 后续不必再找它
                    if self.wait_for_pick.count(head_point_index) > 0:  # 这个点下标在wait_for_pick队列中
                        self.wait_for_pick.pop(self.wait_for_pick.index(head_point_index))  # 那就pop掉
                    self.points[head_point_index].setClusterNum(self.cur_cluster_num)  # 在这个seeds里能找到， 说明簇号还是一样
                    if self.points[head_point_index].getPointType() == UNVISITED:  # 还没标记过
                        cur_near_pts = self.findNearPoint(head_point_index)  # 查看一下它的邻近点
                        if len(cur_near_pts) > self.min_pts:  # 如果邻近点数量超过阈值
                            self.points[head_point_index].setPointType(CORE)  # 核心点
                            # 把邻近点全部加入seeds
                            for p in cur_near_pts:
                                if p not in pre_points:  # 没遍历过的邻近点才加入
                                    seeds.put(p)
                        else:  # 邻近点数量没到
                            self.points[head_point_index].setPointType(BOARD)  # 边界点
                    else:  # 已经被标记过
                        if self.points[head_point_index].getPointType() == NOISE:  # 之前是噪声
                            self.points[head_point_index].setPointType(BOARD)  # 现在改成边界点
                self.cur_cluster_num = self.cur_cluster_num + 1  # 下一个簇号

    def getAllCluster(self):
        """
        :return: 返回所有点的标签
        """
        return [self.points[i].getClusterNum() + 1 for i in range(len(self.points))]
        # +1 是为了防止标签为-1的点没有对应的颜色

    def printPointAndLabel(self):
        """
        这个方法打印点信息，主要用来测试
        :return: 无返回值
        """
        cluster_list = [[] for i in range(self.cur_cluster_num)]
        for pot in range(len(self.points)):
            print(self.points[pot].getClusterNum(), self.cur_cluster_num)
            if self.points[pot].getClusterNum() > -1:
                cluster_list[self.points[pot].getClusterNum()].append(pot)
        for li in range(len(cluster_list)):
            if len(cluster_list[li]) > 0:
                print('cluster #{0}'.format(li))
                for i in cluster_list[li]:
                    print('({0}, {1})'.format(self.points[i].x, self.points[i].y))
                print('--------------')


def dbScanCluster(total_size=200, xScale=1000, yScale=1000, eps=80, min_pts=3):
    # 目前只支持生成随机点
    x, y = generate(total=total_size, x_scale=xScale, y_scale=yScale)  # 生成随机点
    db = dbscan_cluster(x, y, eps, min_pts)
    db.mainProcess()
    label = db.getAllCluster()
    maxCluster = db.getClusterNum()
    return x, y, label, maxCluster
