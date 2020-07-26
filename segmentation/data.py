import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

N_CPU = 2 if torch.cuda.is_available() else 0
H, W = 240, 320


def get_train_dataloader(dataset_file, batch_size):
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(10, fill=(0,)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_dataset = HandDataset(dataset_file, transform)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=N_CPU)
    return train_dataloader


def get_eval_dataloader(dataset_file, batch_size):
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_dataset = HandDataset(dataset_file, transform)
    eval_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=N_CPU)
    return eval_dataloader


def get_img(img_path, device):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    return img


def get_original_img(img_path):
    original_image = Image.open(img_path).convert('RGB').resize((W, H))
    return original_image


class HandDataset(Dataset):
    def __init__(self, dataset_file, transform=None):
        super(HandDataset, self).__init__()
        with open(dataset_file, 'r') as f:
            self.filenames = f.readlines()
        self.transform = transform

    def __getitem__(self, idx):
        img_path, label = self.filenames[idx].split()[0], int(self.filenames[idx].split()[1])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, 'label': label, 'img_path': img_path}

        return sample

    def __len__(self):
        return len(self.filenames)
