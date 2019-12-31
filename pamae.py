import time
import multiprocessing
from multiprocessing import cpu_count
import random
from kmedoids import pam
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import pickle
import argparse
import os


def setup_args():
    """
    设置参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', default=10000, type=int, required=False, help='生成的数据的个数')
    parser.add_argument('--subset_size', default=100, type=int, required=False, help='phase 1中采样子集的规模')
    parser.add_argument('--subset_num', default=5, type=int, required=False, help='phase 1中采样子集的数量')
    parser.add_argument('--centroid_num', default=10, type=int, required=False, help='簇中心的数量')
    return parser.parse_args()


args = setup_args()


def euclid_distance(x, y):
    # 欧式距离
    return np.sqrt(sum(np.square(x - y)))


def generate_data(n_points, centroid_num, n_features):
    """
    生成数据
    :param n_points: 生成数据的数量
    :param centroid_num: 生成数据的中心点数量
    :param n_features: 数据维度
    :return:
    """
    data, target = make_blobs(n_samples=n_points, n_features=n_features, centers=centroid_num)
    # 添加噪声
    np.put(data, [n_points, 0], 10, mode='clip')
    np.put(data, [n_points, 1], 10, mode='clip')
    # 画图
    # pyplot.scatter(data[:, 0], data[:, 1], c=target)
    # pyplot.title("generate data")
    # pyplot.show()
    with open("data", "wb") as f:
        pickle.dump(data.tolist(), f)
    return data.tolist()


def assign_points(data, centroids):
    """
    将所有数据点划分到距离最近的中心
    :param data:数据集
    :param centroids:中心点集合
    :return:
    """
    cluster_points = [[centroid] for centroid in centroids]
    labels = []
    distances_sum = 0  # 聚类误差
    for point in data:
        # 计算节点point到每个中心的距离，并将其划分到最近的中心点
        distances = [euclid_distance(point, centroid) for centroid in centroids]
        label = np.argmin(distances)  # 选择距离最近的簇中心
        labels.append(label)
        distances_sum += distances[label]
        cluster_points[label].append(point)  # 将point加入距离最近的簇中
    return labels, cluster_points, distances_sum


def sampling_and_clustering(data, n_samples, centroid_num):
    """
    对data进行随机采样，并且进行聚类
    :param data:数据集
    :param n_samples:每个子集的数据点的数量
    :param centroid_num:中心点的数量
    :return:
    """
    if n_samples > len(data):
        return data
    subset = random.sample(data, n_samples)
    subset = np.array(subset)
    # 对随机采样的子集进行聚类，获得子集的中心集合centroids
    centroids, _, _ = pam(subset, centroid_num)
    # 将Entire data的点划分到最近的中心，计算聚类误差
    labels, cluster_points, distances_sum = assign_points(data, centroids)
    return centroids, labels, distances_sum


def search_centroid(data):
    """
    找到数据集的中心
    :param data:数据集
    :return:
    """
    min_distances_sum = float("Inf")
    centroid = None
    # 尝试让第i个簇的每个非中心点替换中心点，若聚类误差降低，则交换
    for point in data:
        # 计算每个节点到假定中心的距离
        distances = [euclid_distance(point_1, point) for point_1 in data]
        distances_sum = sum(distances)
        # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
        if distances_sum < min_distances_sum:
            min_distances_sum = distances_sum
            centroid = point
    return centroid


def phase1(data, subset_size, subset_num, centroid_num, pool):
    """
    第一阶段的算法
    :param data: 数据集
    :param subset_size: 采样后子集的大小
    :param subset_num: 子集数量
    :param centroid_num: 簇中心数量
    :return:
    """
    start = time.perf_counter()  # 开始计时
    results = []
    for i in range(subset_num):
        result = pool.apply_async(sampling_and_clustering, (data, subset_size, centroid_num))  # 异步并行计算
        results.append(result)

    min_distancec_sum = float('inf')
    best_labels = None
    best_centroids = None

    # 选择聚类误差最小的medoids
    for i in range(0, subset_num):
        centroids, labels, distances_sum = results[i].get()
        if distances_sum < min_distancec_sum:
            min_distancec_sum = distances_sum
            best_centroids = centroids
            best_labels = labels
    end = time.perf_counter()  # 计时结束
    phase1_time = end - start  # 耗费时间phase1 消耗的时间
    print("PHASE 1运行结束，运行时间：{}秒".format(phase1_time))
    print("PHASE 1得到的候选中心集合为：")
    for centroid in best_centroids:
        print(centroid)
    print("聚类误差:{}".format(min_distancec_sum))
    return best_centroids, best_labels, min_distancec_sum, phase1_time


def phase2(data, centroids, centroid_num, pool):
    start = time.perf_counter()  # 开始计时

    # 将Entire data的点划分到距离最近的中心
    labels, cluster_points, _ = assign_points(data, centroids)

    # 并行更新每个簇的中心
    results = []
    for i in range(centroid_num):
        result = pool.apply_async(search_centroid, (cluster_points[i],))  # 异步并行计算
        results.append(result)
    new_centroids = [result.get() for result in results]

    # 将数据划分到新的簇中心
    labels, cluster_points, distances_sum = assign_points(data, new_centroids)
    end = time.perf_counter()  # 计时结束
    phase2_time = end - start  # 耗费时间phase2 消耗的时间
    print("PHASE 2运行结束，运行时间：{}秒".format(phase2_time))
    print("PHASE 2得到的中心集合为：")
    for centroid in new_centroids:
        print(centroid)
    print("聚类误差:{}".format(distances_sum))
    return new_centroids, labels, distances_sum, phase2_time


def draw_scatter(title, x, y, centroids, labels, n_points, subset_size, subset_num, centroid_num):
    pyplot.scatter(x, y, c=labels)
    centroids_x = []
    centroids_y = []
    for centroid in centroids:
        centroids_x.append(centroid[0])
        centroids_y.append(centroid[1])
    pyplot.scatter(centroids_x, centroids_y, c="r", marker="p")
    pyplot.title(title)
    # phase1_数据集大小_采样子集大小_采样子集数量_中心数量
    path = "results"
    if not os.path.exists(path):
        os.mkdir(path)
    pyplot.savefig(
        "{}/{}_{}_{}_{}_{}.png".format(path, title, n_points, subset_size, subset_num, centroid_num))
    pyplot.show()


def main():
    n_points = args.n_points  # 生成的数据的个数
    subset_size = args.subset_size  # phase 1中采样后子集的大小
    subset_num = args.subset_num  # phase 1中采样的子集的数量
    centroid_num = args.centroid_num  # 簇中心的数量
    n_features = 2  # 数据维度
    # with open("data", "rb") as f:
    #     data = pickle.load(f)
    data = generate_data(n_points, centroid_num, n_features)  # 生成数据
    pool = multiprocessing.Pool(processes=cpu_count())  # 进程池

    ##########################
    #   PHASE 1
    ##########################
    print("正在运行PHASE 1...")
    centroids, labels, distances_sum, phase1_time = phase1(data, subset_size, subset_num, centroid_num, pool)
    data = np.array(data)
    draw_scatter("PHASE1", data[:, 0], data[:, 1], centroids, labels, n_points, subset_size, subset_num, centroid_num)

    ##########################
    #   PHASE 2
    ##########################
    print("正在运行PHASE 2...")
    centroids, labels, distances_sum, phase2_time = phase2(data, centroids, centroid_num, pool)
    draw_scatter("PHASE2", data[:, 0], data[:, 1], centroids, labels, n_points, subset_size, subset_num, centroid_num)
    print("算法总耗时{}".format(phase1_time + phase2_time))

    pool.close()  # 关闭进程池，不再向进程里增加新的子进程
    pool.join()  # 等待所有进程运行完毕后退出。


if __name__ == '__main__':
    main()
