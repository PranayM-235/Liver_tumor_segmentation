import os
import cv2
import torch
from torch.utils.data import Dataset

class LiverDataset(Dataset):
    def __init__(self, data, base_path):
        self.data = data
        self.base_path = base_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        img = cv2.imread(os.path.join(self.base_path, img_path), 0)
        mask = cv2.imread(os.path.join(self.base_path, mask_path), 0)

        img = cv2.resize(img, (128,128))
        mask = cv2.resize(mask, (128,128))

        img = img / 255.0
        mask = (mask > 0).astype("float32")

        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask