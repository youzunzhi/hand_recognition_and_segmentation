import os, time
import numpy as np
import torch
from PIL import Image
from score_cam import get_binary_cam
from cluster_methods.kernel_k_means import kernel_k_means
from cluster_methods.spectral_clustering import spectral_clustering
from model import get_model
from utils import save_cluster_result_on_image
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def model_select_02():
    valid_img_path_list = ['dataset/人手5/Pic_2018_07_24_101712_blockId#30106.bmp',
                           'dataset/人手3/Pic_2018_07_24_163320_blockId#26002.bmp',
                           'dataset/人手11/Pic_2018_07_24_105705_blockId#27600.bmp',
                           'dataset/人手10/Pic_2018_07_24_105230_blockId#10392.bmp']
    model_name_list = ['resnet18']
    gamma_s_list = [0.0005, 0.00001, 0.000001]
    gamma_c_list = [0.0005, 0.00001, 0.000001]
    cluster_method_list = ['normal_cut']
    for model_name in model_name_list:
        model = get_model(device, model_name, f'outputs/{model_name}/model_saved.pth')
        for img_path in valid_img_path_list:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            binary_cam = get_binary_cam(model, model_name, img_path)
            binary_cam_save_fname = f'outputs/{img_name}/binary_cam/{model_name}.png'
            os.makedirs(f'outputs/{img_name}/binary_cam/', exist_ok=True)
            save_cluster_result_on_image(img_path, binary_cam, binary_cam_save_fname)
            for gamma_s in gamma_s_list:
                for gamma_c in gamma_c_list:
                    for cluster_method in cluster_method_list:
                        save_fname = f'outputs/{img_name}/{model_name}_{cluster_method[0]}_{gamma_s}_{gamma_c}.png'
                        if cluster_method == 'kernel_k_means':
                            cluster_result = kernel_k_means(img_path, gamma_s, gamma_c, binary_cam)
                        elif cluster_method == 'ratio_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, False, binary_cam)
                        elif cluster_method == 'normal_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, True, binary_cam)
                        else:
                            raise NotImplementedError
                        save_cluster_result_on_image(img_path, cluster_result, save_fname)


def model_select_01():
    valid_img_path_list = ['dataset/人手5/Pic_2018_07_24_101712_blockId#30106.bmp',
                           'dataset/人手3/Pic_2018_07_24_163320_blockId#26002.bmp',
                           'dataset/人手11/Pic_2018_07_24_105705_blockId#27600.bmp',
                           'dataset/人手10/Pic_2018_07_24_105230_blockId#10392.bmp']
    # valid_img_path_list = ['dataset/人手2/Pic_2018_07_25_094726_blockId#40724.bmp', 'dataset/人手9/Pic_2018_07_25_104021_blockId#10800.bmp']
    model_name_list = ['resnet18', 'resnet50']
    gamma_s_list = [0.1, 0.05, 0.01, 0.001]
    gamma_c_list = [0.1, 0.05, 0.01, 0.001]
    cluster_method_list = ['kernel_k_means', 'ratio_cut', 'normal_cut']
    for model_name in model_name_list:
        model = get_model(device, model_name, f'outputs/{model_name}/model_saved.pth')
        for img_path in valid_img_path_list:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            binary_cam = get_binary_cam(model, model_name, img_path)
            binary_cam_save_fname = f'outputs/{img_name}/binary_cam/{model_name}.png'
            os.makedirs(f'outputs/{img_name}/binary_cam/', exist_ok=True)
            save_cluster_result_on_image(img_path, binary_cam, binary_cam_save_fname)
            for gamma_s in gamma_s_list:
                for gamma_c in gamma_c_list:
                    for cluster_method in cluster_method_list:
                        save_fname = f'outputs/{img_name}/{model_name}_{cluster_method[0]}_{gamma_s}_{gamma_c}.png'
                        if cluster_method == 'kernel_k_means':
                            cluster_result = kernel_k_means(img_path, gamma_s, gamma_c, binary_cam)
                        elif cluster_method == 'ratio_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, False, binary_cam)
                        elif cluster_method == 'normal_cut':
                            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, True, binary_cam)
                        else:
                            raise NotImplementedError
                        save_cluster_result_on_image(img_path, cluster_result, save_fname)


def output_test_result():
    test_img_name_list = get_test_img_name_list()
    model_name = 'resnet18'
    gamma_s, gamma_c = 0.00001, 0.00001
    cluster_method = 'normal_cut'
    model = get_model(device, model_name, f'outputs/{model_name}/model_saved.pth')
    start_time = time.time()
    for img_name in test_img_name_list:
        img_path = os.path.join('dataset/test data', img_name)
        save_fname = f'outputs/test_result/with_img/{img_name}_with.png'
        os.makedirs('outputs/test_result/with_img', exist_ok=True)
        binary_cam = get_binary_cam(model, model_name, img_path)
        if cluster_method == 'kernel_k_means':
            cluster_result = kernel_k_means(img_path, gamma_s, gamma_c, binary_cam)
        elif cluster_method == 'ratio_cut':
            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, False, binary_cam, save_eigen=False)
        elif cluster_method == 'normal_cut':
            cluster_result = spectral_clustering(img_path, gamma_s, gamma_c, True, binary_cam, save_eigen=False)
        else:
            raise NotImplementedError
        if cluster_result.sum() > cluster_result.size / 2:
            cluster_result = ~cluster_result
        save_cluster_result_on_image(img_path, cluster_result, save_fname)
        save_mask_result(cluster_result, img_name)
        print(f'finish testing {img_name}, time: {time.time()-start_time}s')
        start_time = time.time()


def save_mask_result(cluster_result, img_name):
    mask_result = np.zeros(cluster_result.shape)
    mask_result[cluster_result] = 255.
    mask_result = Image.fromarray(mask_result).resize((640, 480)).convert('L')
    mask_result.save(f"outputs/test_result/{img_name}_mask.bmp")


def get_test_img_name_list():
    test_dataset_root = 'dataset/test data'
    test_img_name_list = []
    for img_name in os.listdir(test_dataset_root):
        if img_name.find('bmp') == -1:
            continue
        test_img_name_list.append(img_name)
    return test_img_name_list


if __name__ == '__main__':
    # model_select_01()
    # model_select_02()
    output_test_result()