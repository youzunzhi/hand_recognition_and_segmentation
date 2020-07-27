import os
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn import svm
from utils import make_data_for_svm, setup_logger, log_info

import random
random.seed(0)
H, W = 60, 80
NUM_K_FOLD = 5


def main():
    X_train, Y_train = make_data_for_svm(H, W, is_training=True)
    best_clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
    best_clf.fit(X_train, Y_train)

    X_test, Y_test = make_data_for_svm(H, W, is_training=False)
    pred = best_clf.predict(X_test)
    TP, FP, TN, FN = 0, 0, 0, 0
    TP += np.logical_and(pred == Y_test, pred == 1).sum()
    FP += np.logical_and(pred != Y_test, pred == 1).sum()
    TN += np.logical_and(pred == Y_test, pred == 0).sum()
    FN += np.logical_and(pred != Y_test, pred == 0).sum()
    print(f"Acc: {(TP+TN)/(TP+FP+TN+FN):.4f} ({TP+TN}/{TP+FP+TN+FN})")
    print(f"Precision: {(TP)/(TP+FP):.4f} ({TP}/{TP+FP})")
    print(f"Recall: {(TP)/(TP+FN):.4f} ({TP}/{TP+FN})")


def grid_search_1():
    setup_logger('outputs/svm_grid_search/grid_search_1.txt')
    X_train, Y_train = make_data_for_svm(H, W, is_training=True)

    # --------- grid search ---------
    best_acc, best_model = 0, ''
    start_time = time.time()
    for kernel in ['linear', 'poly', 'rbf']:
        for C in [0.1, 1, 10]:
            if kernel == 'linear':
                clf = svm.SVC(kernel=kernel, C=C)
                scores = cross_val_score(clf, X_train, Y_train, cv=NUM_K_FOLD)
                log_info(
                    f"kernel:{kernel}, C:{C}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                if best_acc < scores.mean():
                    best_acc = scores.mean()
                    best_model = f"kernel:{kernel}, C:{C}"
                start_time = time.time()
            elif kernel == 'poly':
                for gamma in ['scale', 'auto']:
                    for degree in [2, 3, 4]:
                        for coef0 in [0, 0.5, 1]:
                            clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
                            scores = cross_val_score(clf, X_train, Y_train, cv=NUM_K_FOLD)
                            log_info(
                                f"kernel:{kernel}, C:{C}, gamma:{gamma}, degree:{degree}, coef0:{coef0}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                            if best_acc < scores.mean():
                                best_acc = scores.mean()
                                best_model = f"kernel:{kernel}, C:{C}, gamma:{gamma}, degree:{degree}, coef0:{coef0}"
                            start_time = time.time()
            else:
                for gamma in ['scale', 'auto']:
                    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
                    scores = cross_val_score(clf, X_train, Y_train, cv=NUM_K_FOLD)
                    log_info(
                        f"kernel:{kernel}, C:{C}, gamma:{gamma}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                    if best_acc < scores.mean():
                        best_acc = scores.mean()
                        best_model = f"kernel:{kernel}, C:{C}, gamma:{gamma}"
                    start_time = time.time()
    log_info(f"Best model: {best_model}, Acc: {best_acc:.6f}")


def grid_search_2():
    setup_logger('outputs/svm_grid_search/grid_search_2.txt')
    X_train, Y_train = make_data_for_svm(H, W, is_training=True)

    # --------- grid search ---------
    best_acc, best_model = 0, ''
    start_time = time.time()
    for kernel in ['poly', 'rbf']:
        for C in [10, 50, 100, 1000]:
            if kernel == 'poly':
                clf = svm.SVC(kernel=kernel, C=C, gamma='scale', degree=3, coef0=0.5)
                scores = cross_val_score(clf, X_train, Y_train, cv=NUM_K_FOLD)
                log_info(f"kernel:{kernel}, C:{C}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                if best_acc < scores.mean():
                    best_acc = scores.mean()
                    best_model = f"kernel:{kernel}, C:{C}"
                start_time = time.time()
            else:
                clf = svm.SVC(kernel=kernel, C=C, gamma='scale')
                scores = cross_val_score(clf, X_train, Y_train, cv=NUM_K_FOLD)
                log_info(f"kernel:{kernel}, C:{C}, Acc: {scores.mean():.3f} (+/- {scores.std() * 2:.2f}), Time:{time.time() - start_time:.1f}")
                if best_acc < scores.mean():
                    best_acc = scores.mean()
                    best_model = f"kernel:{kernel}, C:{C}"
                start_time = time.time()
    log_info(f"Best model: {best_model}, Acc: {best_acc:.6f}")


if __name__ == '__main__':
    # grid_search_1()
    # grid_search_2()
    # --------------------
    main()