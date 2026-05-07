import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import LiverDataset
from model import UNet

# paths
BASE_PATH = "YOUR_PATH"
CSV_PATH = "YOUR_CSV"

# data
df = pd.read_csv(CSV_PATH)
data = list(zip(df["image"], df["mask"]))

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_loader = DataLoader(LiverDataset(train_data, BASE_PATH), batch_size=8, shuffle=True)
val_loader   = DataLoader(LiverDataset(val_data, BASE_PATH), batch_size=8)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = UNet().to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# training
epochs = 25

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# save
torch.save(model.state_dict(), "UNet_model.pth")