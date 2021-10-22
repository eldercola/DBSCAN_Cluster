from dbscan_cluster import dbScanCluster
from config import scatterDifferent


if __name__ == '__main__':
    x, y, label, maxCluster = dbScanCluster(total_size=100, eps=100, min_pts=2)
    scatterDifferent(x, y, label, maxCluster, color_list=['#8E05C2', '#A9333A', '#3E7C17', 'blue', '#F4A442', '#FF9292', '#1DB9C3', '#6D9886'], marker_list=['^', 'o', '1', 'p', 's'])
