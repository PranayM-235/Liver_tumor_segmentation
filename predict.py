import cv2
import torch
import numpy as np

def predict(image, model, device):
    img = np.array(image.convert("L"))
    img = cv2.resize(img, (128,128))

    original = img.copy()
    img = img / 255.0

    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(img_tensor)

    prob = torch.sigmoid(output).cpu().squeeze().numpy()
    pred = (prob > 0.2).astype("float32")

    return original, prob, pred