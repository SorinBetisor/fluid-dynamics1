import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class FluidDataset(Dataset):
    def __init__(self, path="ai_data"):
        self.files = sorted(glob.glob(f"{path}/u_t*.csv"))
        if len(self.files) == 0:
            raise RuntimeError("No training files found in 'ai_data/'.")
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        u = pd.read_csv(self.files[idx], header=None).values
        v = pd.read_csv(self.files[idx].replace("u_", "v_"), header=None).values
        w = pd.read_csv(self.files[idx].replace("u_", "w_"), header=None).values

        u = (u - np.mean(u)) / (np.std(u) + 1e-5)
        v = (v - np.mean(v)) / (np.std(v) + 1e-5)
        w = (w - np.mean(w)) / (np.std(w) + 1e-5)

        input = np.stack([u, v], axis=0)
        return torch.tensor(input, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)

class FluidNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train():
    dataset = FluidDataset("ai_data")
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8)

    model = FluidNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    patience, wait = 5, 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        model.eval()
        val_loss = np.mean([loss_fn(model(x), y).item() for x, y in val_dl])

        print(f"Train Loss: {running_loss/len(train_dl):.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "fluid_net_best.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    print("Training complete. Best model saved to 'fluid_net_best.pt'.")

    # Export to TorchScript
    example_input = torch.randn(1, 2, *dataset[0][0].shape[1:])
    scripted_model = torch.jit.trace(model, example_input)
    scripted_model.save("fluid_net_script.pt")
    print("Scripted model saved to 'fluid_net_script.pt'.")

def predict(u, v, model_path="fluid_net_best.pt"):
    assert u.shape == v.shape, "Shapes of u and v must match"
    input_np = np.stack([u, v], axis=0)
    input_np = (input_np - input_np.mean()) / (input_np.std() + 1e-5)
    input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0)

    model = FluidNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        out = model(input_tensor)
        return out.squeeze(0).numpy()

if __name__ == "__main__":
    os.makedirs("ai_data", exist_ok=True)
    train()
