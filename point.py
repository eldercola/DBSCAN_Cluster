from config import *


class point:
    def __init__(self, x_pos, y_pos):
        self.x = x_pos
        self.y = y_pos
        self.point_type = UNVISITED  # 未被访问过
        self.cluster = NONE  # 还不属于任何簇

    @property
    def x_pos(self):
        return self.x

    @property
    def y_pos(self):
        return self.y

    def setPointType(self, t):
        self.point_type = t

    def getPointType(self):
        return self.point_type

    def setClusterNum(self, num):
        self.cluster = num

    def getClusterNum(self):
        return self.cluster

    def calculateDistanceSquare(self, another_point):
        return (self.x - another_point.x)**2 + (self.y - another_point.y)**2
