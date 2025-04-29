import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
import os
from neuralNetwork import FluidPDEAI  # Import your model

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
import os

class FluidDataset(Dataset):
    """
    PyTorch Dataset class for loading fluid simulation data from CSVs.
    Each sample consists of (u, v, w) at time t (input) and (u, v, w) at time t+Δt (target).
    """

    def __init__(self, data_dir="../ai_data"):
        """
        Args:
            data_dir (str): Directory where u_t*, v_t*, w_t* CSV files are stored.
        """
        super(FluidDataset, self).__init__()
        self.u_files = sorted(glob.glob(os.path.join(data_dir, "u_t*.csv")))

        if len(self.u_files) < 2:
            raise RuntimeError(f"Not enough data found in {data_dir}. Need at least 2 time steps.")

    def __len__(self):
        # Number of samples = number of timesteps - 1
        return len(self.u_files) - 1

    def __getitem__(self, idx):
        """
        Returns:
            input_tensor (Tensor): Tensor of shape (3, H, W) - u, v, w at time t
            target_tensor (Tensor): Tensor of shape (3, H, W) - u, v, w at time t+Δt
        """
        # Load input fields at timestep t
        u0 = pd.read_csv(self.u_files[idx], header=None).values
        v0 = pd.read_csv(self.u_files[idx].replace("u_", "v_"), header=None).values
        w0 = pd.read_csv(self.u_files[idx].replace("u_", "w_"), header=None).values

        # Load target fields at timestep t+1
        u1 = pd.read_csv(self.u_files[idx + 1], header=None).values
        v1 = pd.read_csv(self.u_files[idx + 1].replace("u_", "v_"), header=None).values
        w1 = pd.read_csv(self.u_files[idx + 1].replace("u_", "w_"), header=None).values

        input_fields = np.stack([u0, v0, w0], axis=0)  # (3, H, W)
        target_fields = np.stack([u1, v1, w1], axis=0)  # (3, H, W)

        input_tensor = torch.tensor(input_fields, dtype=torch.float32)
        target_tensor = torch.tensor(target_fields, dtype=torch.float32)

        return input_tensor, target_tensor

def train_model(
    data_dir="ai_data",
    model_save_path="NN/fluid_pde_ai.pt",
    in_channels=3,
    out_channels=3,
    batch_size=4,
    learning_rate=0.01,
    epochs=80
):
    """
    Trains the FluidPDEAI model on simulation data.

    Args:
        data_dir (str): Directory containing ai_data CSVs.
        model_save_path (str): Path to save the trained model.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
    """
    # Load dataset
    dataset = FluidDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = FluidPDEAI(in_channels=in_channels, out_channels=out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            preds = model(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # Save trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()