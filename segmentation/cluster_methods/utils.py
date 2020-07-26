import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# H, W = 60, 80
H, W = 30, 40
N = H*W
K = 2


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


def get_init_cluster(binary_cam):
    resized_binary_cam = np.array(Image.fromarray(binary_cam).resize((W, H), Image.NEAREST))
    init_cluster = np.zeros((N, K), dtype=bool)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if resized_binary_cam[i, j]:
                init_cluster[idx, 1] = True
            else:
                init_cluster[idx, 0] = True
    return init_cluster

def read_image(img_path):
    img = Image.open(img_path).resize((W, H))
    img = np.asarray(img)

    data_points = img_to_data_points(img)
    return data_points


def img_to_data_points(img):
    """
    turn an image into a data point matrix
    :param img: shape: H, W
    :return: data_points: shape: H*W, 3(coord_i, coord_j, grayscale)
    """
    data_points = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            data_points.append([i, j, img[i, j]])
    data_points = np.asarray(data_points)
    return data_points
