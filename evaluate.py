import torch
import cv2
import os

def evaluate(model, val_data, BASE_PATH, device):
    TP = TN = FP = FN = 0

    model.eval()

    for img_path, mask_path in val_data:

        img = cv2.imread(os.path.join(BASE_PATH, img_path), 0)
        img = cv2.resize(img, (128,128)) / 255.0

        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(img_tensor)

        prob = torch.sigmoid(output).cpu().squeeze().numpy()
        pred = (prob > 0.2).astype("float32")

        target = cv2.imread(os.path.join(BASE_PATH, mask_path), 0)
        target = cv2.resize(target, (128,128))
        target = (target > 0).astype("float32")

        TP += ((pred == 1) & (target == 1)).sum()
        TN += ((pred == 0) & (target == 0)).sum()
        FP += ((pred == 1) & (target == 0)).sum()
        FN += ((pred == 0) & (target == 1)).sum()

    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)