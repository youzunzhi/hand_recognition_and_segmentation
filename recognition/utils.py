import os, sys, logging
import numpy as np
from PIL import Image, ImageOps


def setup_logger(output_dir, distributed_rank=0):
    # ---- make output dir ----
    os.makedirs(output_dir, exist_ok=True)

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

    txt_name = 'log.txt'
    fh = logging.FileHandler(os.path.join(output_dir, txt_name), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_info(log_str):
    logging.getLogger().info(log_str)


# ---- functions used separately ----

def flip_negative_data_samples():
    with open('dataset/dataset_whole_imbalance.txt', 'r') as f:
        l = f.readlines()
    for sample in l:
        img_path, label = sample.split()[0], int(sample.split()[1])
        if label == 0:
            img = Image.open(img_path)
            img_hflip = ImageOps.flip(img)
            img_vflip = ImageOps.mirror(img)
            img_hflip.save(img_path.replace('.bmp', '-hf.bmp'))
            img_vflip.save(img_path.replace('.bmp', '-vf.bmp'))


def make_dataset_txt():
    pos_cnt, neg_cnt = 0, 0
    with open('dataset_whole.txt', 'w') as f:
        dataset_root = 'dataset'
        for d in os.listdir(dataset_root):
            if d == '.DS_Store' or d.find('.txt') != -1:
                continue
            for img_name in os.listdir(os.path.join(dataset_root, d)):
                if d.find('.bmp') == -1:
                    continue
                label = int(d.find('with_hand') != -1)
                if img_name.find('Pic_2018_07_25_095017_blockId#26365') != -1 or \
                        img_name.find('Pic_2018_07_25_094845_blockId#64085') != -1 or \
                        img_name.find('Pic_2018_07_25_095016_blockId#25912') != -1 or \
                        img_name.find('Pic_2018_07_25_095017_blockId#26214') != -1 or \
                        img_name.find('Pic_2018_07_25_095016_blockId#26063') != -1:
                    label = 0
                if label == 1:
                    pos_cnt += 1
                else:
                    neg_cnt += 1
                f.write(f'{os.path.join(os.path.join(dataset_root, d), img_name)} {label}\n')
    print(pos_cnt, neg_cnt)  # 3353 3093 (3358 1026)
    return pos_cnt, neg_cnt


def make_test_dataset_txt():
    pos_cnt, neg_cnt = 0, 0
    with open('testing_dataset.txt', 'w') as f:
        dataset_root = 'dataset/test_data'
        for d in os.listdir(dataset_root):
            if d == '.DS_Store' or d.find('.txt') != -1:
                continue
            for img_name in os.listdir(os.path.join(dataset_root, d)):
                if d.find('.bmp') == -1:
                    continue
                label = int(d.find('with_hand') != -1)
                if label == 1:
                    pos_cnt += 1
                else:
                    neg_cnt += 1
                f.write(f'{os.path.join(os.path.join(dataset_root, d), img_name)} {label}\n')
    print(pos_cnt, neg_cnt)  # 3353 3093 (3358 1026)
    return pos_cnt, neg_cnt


if __name__ == '__main__':
    # flip_negative_data_samples()
    make_test_dataset_txt()
