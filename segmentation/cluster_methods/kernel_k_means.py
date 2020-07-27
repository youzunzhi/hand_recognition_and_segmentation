import os
import numpy as np
from .utils import read_image, get_init_cluster, compute_gram_matrix, N, K, H, W


def kernel_k_means(img_path, gamma_s, gamma_c, binary_cam):
    data_points = read_image(img_path)
    gram_matrix = compute_gram_matrix(data_points, gamma_s, gamma_c)
    prev_cluster = get_init_cluster(binary_cam)

    for i in range(500):
        new_cluster = get_new_cluster(gram_matrix, prev_cluster)
        if (prev_cluster ^ new_cluster).sum() == 0:
            break
        prev_cluster = new_cluster
    cluster_result = np.zeros((H, W), dtype=bool)
    for i, data_point in enumerate(data_points):
        cluster_result[data_point[0], data_point[1]] = bool(np.argmax(new_cluster[i]))
    return cluster_result


def get_new_cluster(gram_matrix, prev_cluster):
    distance_matrix = get_distance_matrix(gram_matrix, prev_cluster)
    min_ind = np.argmin(distance_matrix, axis=1)
    new_cluster = np.zeros(prev_cluster.shape, dtype=bool)
    new_cluster[np.arange(N), min_ind] = True
    return new_cluster


def get_distance_matrix(gram_matrix, cluster):
    """

    :param gram_matrix:
    :param cluster:
    :return: distance_matrix: shaped (N, K) and the (j, k)-th element is the distance between x_j and the mean of cluster k
    """
    distance_matrix = np.ndarray((N, K))
    for k in range(K):
        term2 = gram_matrix[:, cluster[:, k]].sum(axis=1)
        term3 = gram_matrix[cluster[:, k]][:, cluster[:, k]].sum()
        cluster_k_cnt = cluster[:, k].sum()
        if cluster_k_cnt > 0:
            term2 *= (-2) / cluster_k_cnt
            term3 *= 1 / (cluster_k_cnt ** 2)
        distance_matrix[:, k] = term2 + term3
    return distance_matrix


