import cv2
import os

def get_non_tumor_images(data, BASE_PATH):
    non_tumor_images = []

    for img_path, mask_path in data:
        mask_full = os.path.join(BASE_PATH, mask_path)
        mask = cv2.imread(mask_full, 0)

        if mask is None:
            continue

        if (mask > 0).sum() == 0:
            non_tumor_images.append((img_path, mask_path))

    return non_tumor_images

def dice_score(pred, target):
    pred = (pred > 0.2).astype("float32")

    smooth = 1e-5
    intersection = (pred * target).sum()

    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)