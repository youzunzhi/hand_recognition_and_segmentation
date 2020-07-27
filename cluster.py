import os
import time
import numpy as np
from utils import make_data_for_clustering, save_cluster_result_on_image

np.random.seed(0)
H, W = 30, 40
# H, W = 60, 80
N = H * W
K = 2


def output_segmentation(img_path):
    cluster_method = 'normal_cut'
    init_method = 'kpp'
    gamma_s, gamma_c = 1e-03, 1e-03
    start_time = time.time()
    img_fname = os.path.split(img_path)[1]
    save_fname = f'outputs/segmentation/{img_fname.replace(".bmp", "_seg.png")}'
    os.makedirs('outputs/segmentation', exist_ok=True)

    if cluster_method == 'kernel_k_means':
        cluster_result = kernel_k_means(img_path, gamma_s, gamma_c, init_method)
    elif cluster_method == 'ratio_cut':
        cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, False, init_method,
                                             save_eigen=False)
    elif cluster_method == 'normal_cut':
        cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, True, init_method,
                                             save_eigen=False)
    else:
        raise NotImplementedError
    save_cluster_result_on_image(img_path, cluster_result, save_fname)
    print(f'finish segmenting {img_fname}, time: {time.time() - start_time}s')


def grid_search_1():
    os.makedirs('outputs/seg_grid_search_1/', exist_ok=True)
    valid_img_path_list = ['dataset/11_with_hand/Pic_2018_07_24_105705_blockId#27600.bmp',
                           'dataset/5_with_hand/Pic_2018_07_24_101712_blockId#30106.bmp',
                           'dataset/12_with_hand/Pic_2018_07_25_105137_blockId#17103.bmp',
                           'dataset/18_with_hand/Pic_2018_07_25_141708_blockId#43065.bmp']
    gamma_s_list = [0.1, 0.01, 0.001]
    gamma_c_list = [0.1, 0.01, 0.001]
    cluster_method_list = ['kernel_k_means', 'ratio_cut', 'normal_cut']
    init_method_list = ['random', 'split', 'kpp']
    start_time = time.time()
    for img_path in valid_img_path_list:
        img_fname = os.path.split(img_path)[1]
        img_name = os.path.splitext(img_fname)[0]
        img_num = img_name.split('#')[-1]
        for gamma_s in gamma_s_list:
            for gamma_c in gamma_c_list:
                for cluster_method in cluster_method_list:
                    for init_method in init_method_list:
                        save_fname = f'outputs/seg_grid_search_1/{img_num}_{init_method[0]}_{cluster_method[0]}_{gamma_s}_{gamma_c}.png'
                        if cluster_method == 'kernel_k_means':
                            cluster_result = kernel_k_means(img_path, gamma_s, gamma_c, init_method)
                        elif cluster_method == 'ratio_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, False, init_method,
                                                                 save_eigen=False)
                        elif cluster_method == 'normal_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, True, init_method,
                                                                 save_eigen=False)
                        else:
                            raise NotImplementedError
                        save_cluster_result_on_image(img_path, cluster_result, save_fname)
                        print(f'time: {time.time() - start_time:.03f}s')
                        start_time = time.time()


def grid_search_2():
    os.makedirs('outputs/seg_grid_search_2/', exist_ok=True)
    valid_img_path_list = ['dataset/11_with_hand/Pic_2018_07_24_105705_blockId#27600.bmp',
                           'dataset/5_with_hand/Pic_2018_07_24_101712_blockId#30106.bmp',
                           'dataset/12_with_hand/Pic_2018_07_25_105137_blockId#17103.bmp',
                           'dataset/18_with_hand/Pic_2018_07_25_141708_blockId#43065.bmp']
    gamma_s_list = [0.0001, 0.00005, 0.00001]
    gamma_c_list = [0.0001, 0.00005, 0.00001]
    cluster_method_list = ['ratio_cut', 'normal_cut']
    init_method_list = ['kpp']
    start_time = time.time()
    for img_path in valid_img_path_list:
        img_fname = os.path.split(img_path)[1]
        img_name = os.path.splitext(img_fname)[0]
        img_num = img_name.split('#')[-1]
        for gamma_s in gamma_s_list:
            for gamma_c in gamma_c_list:
                for cluster_method in cluster_method_list:
                    for init_method in init_method_list:
                        save_fname = f'outputs/seg_grid_search_1/{img_num}_{init_method[0]}_{cluster_method[0]}_{gamma_s}_{gamma_c}.png'
                        if cluster_method == 'kernel_k_means':
                            cluster_result = kernel_k_means(img_path, gamma_s, gamma_c, init_method)
                        elif cluster_method == 'ratio_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, False, init_method,
                                                                 save_eigen=False)
                        elif cluster_method == 'normal_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, True, init_method,
                                                                 save_eigen=False)
                        else:
                            raise NotImplementedError
                        save_cluster_result_on_image(img_path, cluster_result, save_fname)
                        print(f'time: {time.time() - start_time:.03f}s')
                        start_time = time.time()


def kernel_k_means(img_path, gamma_s, gamma_c, init_method):
    data_points = make_data_for_clustering(img_path, H, W)
    gram_matrix = compute_gram_matrix(data_points, gamma_s, gamma_c)
    prev_cluster = get_init_cluster(gram_matrix, init_method)

    for i in range(500):
        new_cluster = get_new_cluster(gram_matrix, prev_cluster)
        if (prev_cluster ^ new_cluster).sum() == 0:
            break
        prev_cluster = new_cluster
    cluster_result = np.zeros((H, W), dtype=bool)
    for i, data_point in enumerate(data_points):
        cluster_result[data_point[0], data_point[1]] = bool(np.argmax(new_cluster[i]))
    return cluster_result


def spectral_clustering(img_path, gamma_s, gamma_c, normalize, init_method, save_eigen=True):
    data_points = make_data_for_clustering(img_path, H, W)
    N = len(data_points)
    os.makedirs('outputs/eigens_save', exist_ok=True)
    img_name = os.path.split(img_path)[-1].split('.')[0]
    eigen_values_path = os.path.join('outputs/eigens_save',
                                     f'{img_name}_{"n" if normalize else "r"}_{gamma_s}_{gamma_c}_vals.npy')
    eigen_vectors_path = os.path.join('outputs/eigens_save',
                                      f'{img_name}_{"n" if normalize else "r"}_{gamma_s}_{gamma_c}_vecs.npy')
    if os.path.exists(eigen_values_path):
        print(f'load {eigen_vectors_path} and {eigen_values_path}')
        eigen_values = np.load(eigen_values_path)
        eigen_vectors = np.load(eigen_vectors_path)
    else:
        gram_matrix = compute_gram_matrix(data_points, gamma_s, gamma_c)
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
    cluster_result = k_means_for_spectral_clustering(data_points, U, init_method)
    return cluster_result


def k_means_for_spectral_clustering(data_points, eigenspace_data_points, init_method):
    gram_matrix = 1 - compute_squared_distance(eigenspace_data_points)
    prev_cluster = get_init_cluster(gram_matrix, init_method)
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


def compute_gram_matrix(data_points, gamma_s, gamma_c):
    spatial_data = data_points[:, :2]
    color_data = data_points[:, 2:]
    spatial_squared_dist = compute_squared_distance(spatial_data)
    color_squared_dist = compute_squared_distance(color_data)
    gram_matrix = np.exp(-gamma_s * spatial_squared_dist) * np.exp(-gamma_c * color_squared_dist)
    return gram_matrix


def compute_squared_distance(X):
    """
    actually scipy.spatial.distance.cdist(X, X) ** 2
    :param X: shape: (N, p)
    :return: squared_distance_matrix: (i, j)th element is the squared distance b/w X[i] and X[j]
    """
    X_sq_sum = (X ** 2).sum(axis=1)
    squared_distance = -2 * X.dot(X.T) + X_sq_sum.reshape(-1, 1) + X_sq_sum
    return squared_distance


def get_init_cluster(gram_matrix, init_method):
    init_cluster = np.zeros((N, K), dtype=bool)
    if init_method == 'random':
        random_cluster_choice = np.random.randint(0, K, size=N)
        init_cluster[np.arange(N), random_cluster_choice] = True
    elif init_method == 'split':
        cnt = 0
        for k in range(K - 1):
            init_cluster[cnt:cnt + N // K, k] = True
            cnt += N // K
        init_cluster[cnt:, K - 1] = True
    elif init_method == 'kpp':
        centroids_idx = [np.random.randint(0, N)]
        for _ in range(K - 1):
            dist_to_centroids = gram_matrix[:, centroids_idx]
            if len(centroids_idx) == 1:
                dist_to_nearest_centroid = dist_to_centroids
            else:
                dist_to_nearest_centroid = dist_to_centroids.max(1)
            centroids_idx.append(np.argmin(dist_to_nearest_centroid))
        cluster_choice = gram_matrix[centroids_idx].argmax(0)
        init_cluster[np.arange(N), cluster_choice] = True

    else:
        raise NotImplementedError
    assert init_cluster.sum() == N
    return init_cluster


if __name__ == '__main__':
    # grid_search_1()
    # grid_search_2()
    # --------------------
    img_path = 'dataset/test_data/with_hand/Pic_2018_07_24_100447_blockId#2797.bmp'
    output_segmentation(img_path)
