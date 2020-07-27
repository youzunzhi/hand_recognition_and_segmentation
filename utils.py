import os, sys
import time
import logging
import numpy as np
from PIL import Image

import random

random.seed(0)


def setup_logger(log_txt_path, distributed_rank=0):
    # ---- make output dir ----
    log_txt_dir = os.path.split(log_txt_path)[0]
    os.makedirs(log_txt_dir, exist_ok=True)

    # ---- set up logger ----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", '%m%d%H%M%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_txt_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_info(log_str):
    logging.getLogger().info(log_str)


def make_data_for_svm(img_h, img_w, is_training):
    post_fix = 'train' if is_training else 'test'
    if os.path.exists(f'X_{post_fix}.npy'):
        X = np.load(f'X_{post_fix}.npy')
        Y = np.load(f'Y_{post_fix}.npy')
        return X, Y
    else:
        X, Y = [], []
        dataset_txt_name = 'dataset/training_dataset.txt' if is_training else 'dataset/testing_dataset.txt'
        with open(dataset_txt_name, 'r') as fr:
            dataset_list = fr.readlines()
        if is_training:
            random.shuffle(dataset_list)
        for sample in dataset_list:
            img_path, label = sample.split()[0], int(sample.split()[1])
            Y.append(label)
            img = np.array(Image.open(img_path).resize((img_w, img_h)))
            X.append(img.ravel().tolist())
        X, Y = np.asarray(X), np.asarray(Y)
        np.save(f'X_{post_fix}.npy', X)
        np.save(f'Y_{post_fix}.npy', Y)
        return X, Y


def make_data_for_clustering(img_path, img_h, img_w, ):
    img = Image.open(img_path).resize((img_w, img_h))
    img = np.asarray(img)
    data_points = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            data_points.append([i, j, img[i, j]])
    data_points = np.asarray(data_points)
    return data_points


def save_cluster_result_on_image(img_path, cluster_result, save_fname):
    if cluster_result.sum() > cluster_result.size / 2:
        cluster_result = ~cluster_result
    org_im = Image.open(img_path)
    cluster_result_on_image = apply_cluster_result_on_image(org_im, cluster_result)
    cluster_result_on_image.save(save_fname)
    print('saved to', save_fname)


def apply_cluster_result_on_image(org_im, cluster_result):
    heatmap = np.zeros((cluster_result.shape[0], cluster_result.shape[1], 4))
    heatmap[:, :, 3] = 0.2
    heatmap[cluster_result] = np.array([1., 0, 0, 0.4])
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(org_im.size, Image.NEAREST)

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image
