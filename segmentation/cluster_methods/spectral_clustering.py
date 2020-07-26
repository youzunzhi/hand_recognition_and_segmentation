import os
import numpy as np
from .utils import read_image, compute_squared_distance, compute_gram_matrix, get_init_cluster, K, H, W


def spectral_clustering(img_path, gamma_s, gamma_c, normalize, binary_cam, save_eigen=True):
    data_points = read_image(img_path)
    N = len(data_points)
    os.makedirs('outputs/eigens_save', exist_ok=True)
    img_name = os.path.split(img_path)[-1].split('.')[0]
    eigen_values_path = os.path.join('outputs/eigens_save', f'{img_name}_{"n" if normalize else "r"}_{gamma_s}_{gamma_c}_vals.npy')
    eigen_vectors_path = os.path.join('outputs/eigens_save', f'{img_name}_{"n" if normalize else "r"}_{gamma_s}_{gamma_c}_vecs.npy')
    if os.path.exists(eigen_values_path):
        print(f'load {eigen_vectors_path} and {eigen_values_path}')
        eigen_values = np.load(eigen_values_path)
        eigen_vectors = np.load(eigen_vectors_path)
    else:
        gram_matrix = compute_gram_matrix(data_points, gamma_s, gamma_c)
        # visualize_gram_matrix(gram_matrix)
        degree_matrix = np.eye(N) * gram_matrix.sum(0)
        graph_laplacian_matrix = degree_matrix - gram_matrix
        if normalize:
            degree_matrix_1_2 = np.eye(N) * (gram_matrix.sum(0) ** (-1 / 2))
            graph_laplacian_matrix = (degree_matrix_1_2).dot(graph_laplacian_matrix).dot(degree_matrix_1_2)
        eigen_values, eigen_vectors = np.linalg.eig(graph_laplacian_matrix)
        if save_eigen:
            np.save(eigen_values_path, eigen_values)
            np.save(eigen_vectors_path, eigen_vectors)
            print(f'saved {eigen_vectors_path} and {eigen_values_path}')

    sort_k_idx = np.argsort(eigen_values)[:K]
    U = eigen_vectors[:, sort_k_idx]
    if normalize:
        U_rows_norm = np.linalg.norm(U, ord=2, axis=1)
        U = U / U_rows_norm.reshape(-1, 1)
    cluster_result = k_means(data_points, U, binary_cam)
    return cluster_result


def k_means(data_points, eigenspace_data_points, binary_cam):
    init_cluster = get_init_cluster(binary_cam)
    gram_matrix = 1 - compute_squared_distance(eigenspace_data_points)
    prev_cluster = init_cluster
    for i in range(100):
        new_cluster = get_new_cluster(gram_matrix, prev_cluster)
        if (prev_cluster ^ new_cluster).sum() == 0:
            break
        prev_cluster = new_cluster
    cluster_result = np.zeros((H, W), dtype=bool)
    for i, data_point in enumerate(data_points):
        cluster_result[data_point[0], data_point[1]] = bool(np.argmax(new_cluster[i]))
    return cluster_result


def get_new_cluster(gram_matrix, prev_cluster):
    N = len(gram_matrix)
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
    N = len(gram_matrix)
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




# if __name__ == '__main__':
#     main()

