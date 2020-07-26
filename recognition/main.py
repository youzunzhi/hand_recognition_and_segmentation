import os
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn import svm
from utils import setup_logger, log_info

import random
random.seed(0)
H, W = 60, 80
# H, W = 30, 40
NUM_K_FOLD = 5


def model_select_02():
    OUTPUT_DIR = f'outputs/model_select_02'
    setup_logger(OUTPUT_DIR)

    X, Y = make_data()
    # grid search
    best_acc, best_model = 0, ''
    start_time = time.time()
    for kernel in ['poly', 'rbf']:
        for C in [10, 50, 100, 1000]:
            if kernel == 'poly':
                clf = svm.SVC(kernel=kernel, C=C, gamma='scale', degree=3, coef0=0.5)
                scores = cross_val_score(clf, X, Y, cv=NUM_K_FOLD)
                log_info(f"kernel:{kernel}, C:{C}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                if best_acc < scores.mean():
                    best_acc = scores.mean()
                    best_model = f"kernel:{kernel}, C:{C}"
                start_time = time.time()
            else:
                clf = svm.SVC(kernel=kernel, C=C, gamma='scale')
                scores = cross_val_score(clf, X, Y, cv=NUM_K_FOLD)
                log_info(f"kernel:{kernel}, C:{C}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                if best_acc < scores.mean():
                    best_acc = scores.mean()
                    best_model = f"kernel:{kernel}, C:{C}"
                start_time = time.time()
    log_info(f"Best model: {best_model}, Acc: {best_acc:.6f}")


def model_select_01():
    OUTPUT_DIR = f'outputs/model_select_01'
    setup_logger(OUTPUT_DIR)

    X, Y = make_data()
    # grid search
    best_acc, best_model = 0, ''
    start_time = time.time()
    for kernel in ['linear', 'poly', 'rbf']:
        for C in [0.1, 1, 10]:
            if kernel == 'linear':
                clf = svm.SVC(kernel=kernel, C=C)
                scores = cross_val_score(clf, X, Y, cv=NUM_K_FOLD)
                log_info(f"kernel:{kernel}, C:{C}, Acc: {scores.mean():.3f} (+/- {scores.std()*2:.2f}), Time:{time.time()-start_time:.1f}")
                if best_acc < scores.mean():
                    best_acc = scores.mean()
                    best_model = f"kernel:{kernel}, C:{C}"
                start_time = time.time()
            elif kernel == 'poly':
                for gamma in ['scale', 'auto']:
                    for degree in [2, 3, 4]:
                        for coef0 in [0, 0.5, 1]:
                            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
                            scores = cross_val_score(clf, X, Y, cv=NUM_K_FOLD)
                            log_info(f"kernel:{kernel}, C:{C}, gamma:{gamma}, degree:{degree}, coef0:{coef0}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                            if best_acc < scores.mean():
                                best_acc = scores.mean()
                                best_model = f"kernel:{kernel}, C:{C}, gamma:{gamma}, degree:{degree}, coef0:{coef0}"
                            start_time = time.time()
            else:
                for gamma in ['scale', 'auto']:
                    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
                    scores = cross_val_score(clf, X, Y, cv=NUM_K_FOLD)
                    log_info(f"kernel:{kernel}, C:{C}, gamma:{gamma}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                    if best_acc < scores.mean():
                        best_acc = scores.mean()
                        best_model = f"kernel:{kernel}, C:{C}, gamma:{gamma}"
                    start_time = time.time()
    log_info(f"Best model: {best_model}, Acc: {best_acc:.6f}")


def output_test_result():
    X, Y = make_data()
    best_clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
    best_clf.fit(X, Y)
    img_name_list, X_test = make_test_data()
    test_prediction = best_clf.predict(X_test)
    with open('outputs/test_result.txt', 'w') as f:
        for i in range(len(test_prediction)):
            write_line = f"{img_name_list[i]}    {test_prediction[i]}\n"
            f.write(write_line)

def make_data():
    if os.path.exists('X.npy'):
        X = np.load('X.npy')
        Y = np.load('Y.npy')
        return X, Y
    X, Y = [], []
    with open('dataset/dataset_whole.txt', 'r') as fr:
        l = fr.readlines()
    random.shuffle(l)
    for sample in l:
        img_path, label = sample.split()[0], int(sample.split()[1])
        Y.append(label)
        img = np.array(Image.open(img_path).resize((W, H)))
        X.append(img.ravel().tolist())
    X, Y = np.asarray(X), np.asarray(Y)
    np.save('X', X)
    np.save('Y', Y)
    return X, Y


def make_test_data():
    test_dataset_root = 'dataset/test data'
    img_name_list, X_test = [], []
    for img_name in os.listdir(test_dataset_root):
        if img_name.find('bmp') == -1:
            continue
        img = np.array(Image.open(os.path.join(test_dataset_root, img_name)).resize((W, H)))
        X_test.append(img.ravel().tolist())
        img_name_list.append(img_name)
    X_test = np.asarray(X_test)
    return img_name_list, X_test


if __name__ == '__main__':
    # model_select_01()
    # model_select_02()
    output_test_result()