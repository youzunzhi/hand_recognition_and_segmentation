import os
import torch
import tqdm
import argparse
import torch.nn as nn
import numpy as np
from model import get_model
from data import get_train_dataloader, get_eval_dataloader, get_img
from utils import setup_logger, log_info
from yacs.config import CfgNode as CN
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_cfg():
    cfg = CN()
    cfg.BATCH_SIZE = 32 if torch.cuda.is_available() else 2
    cfg.MODEL_NAME = 'resnet18'
    cfg.LR = 4e-3
    cfg.TOTAL_EPOCHS = 40
    # ---------
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("opts", help="Modify configs using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.merge_from_list(args.opts)
    # ---------
    cfg.EXPERIMENT_NAME = f'{cfg.MODEL_NAME}'
    cfg.OUTPUT_DIR = f'outputs/{cfg.EXPERIMENT_NAME}'
    setup_logger(cfg)
    log_info(cfg)
    return cfg


def train(cfg):
    train_dataloader = get_train_dataloader('dataset/dataset_whole.txt', cfg.BATCH_SIZE)
    model = get_model(device, cfg.MODEL_NAME)
    optimizer = torch.optim.Adam(model.parameters(), cfg.LR, weight_decay=1e-4)
    writer = SummaryWriter(cfg.OUTPUT_DIR)
    for epoch in range(1, cfg.TOTAL_EPOCHS + 1):
        model.train()
        for batch_i, batch in enumerate(train_dataloader):
            imgs = batch['img'].to(device)
            labels = batch['label'].to(device)
            outputs = model(imgs)    # shape: B, 2

            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(1).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            TP = np.logical_and(pred == labels, pred == 1).sum()
            FP = np.logical_and(pred != labels, pred == 1).sum()
            TN = np.logical_and(pred == labels, pred == 0).sum()
            FN = np.logical_and(pred != labels, pred == 0).sum()

            log_info(f"Epoch {epoch}/{cfg.TOTAL_EPOCHS}, Batch {batch_i}/{len(train_dataloader)}, Loss {loss.data}, "
                     f"Acc: {(TP + TN) / (TP + FP + TN + FN):.4f} ({TP + TN}/{TP + FP + TN + FN}), "
                     f"Precision: {(TP) / (TP + FP):.4f} ({TP}/{TP + FP}), "
                     f"Recall: {(TP) / (TP + FN):.4f} ({TP}/{TP + FN})")
            writer.add_scalar('loss', loss.item(), batch_i+(epoch-1)*len(train_dataloader))

        # --- SAVE MODEL ---
        model_save_path = os.path.join(cfg.OUTPUT_DIR, f'model_saved.pth')
        torch.save(model.state_dict(), model_save_path)
        # eval(model=model)


def eval(weight_path=None, model=None):
    eval_dataloader = get_eval_dataloader('dataset/test_dataset.txt', 32)
    if model is None:
        model = get_model(device, weight_path)
    TP, FP, TN, FN = 0, 0, 0, 0
    for batch in tqdm.tqdm(eval_dataloader, desc='EVALUATING'):
        imgs = batch['img'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(imgs)  # shape: B, 2
        pred = outputs.argmax(1).cpu().numpy()
        labels = labels.cpu().numpy()
        TP += np.logical_and(pred == labels, pred == 1).sum()
        FP += np.logical_and(pred != labels, pred == 1).sum()
        TN += np.logical_and(pred == labels, pred == 0).sum()
        FN += np.logical_and(pred != labels, pred == 0).sum()
    log_info(f"Acc: {(TP+TN)/(TP+FP+TN+FN):.4f} ({TP+TN}/{TP+FP+TN+FN})")
    log_info(f"Precision: {(TP)/(TP+FP):.4f} ({TP}/{TP+FP})")
    log_info(f"Recall: {(TP)/(TP+FN):.4f} ({TP}/{TP+FN})")


def pred(img_path, weight_path):
    img = get_img(img_path, device)
    model = get_model(device, weight_path)
    outputs = model(img)
    pred = outputs.argmax(1)
    print(pred)
    return pred


if __name__ == '__main__':
    cfg = get_cfg()
    # -----------
    train(cfg)
    # -----------
    # weight_path = 'outputs/alexnet/model_saved.pth'
    # eval(weight_path=weight_path)
    # -----------
    # img_path = 'dataset/人手3/Pic_2018_07_24_163324_blockId#27358.bmp'
    # weight_path = 'outputs/alexnet/model_saved.pth'
    # pred(img_path, weight_path)
