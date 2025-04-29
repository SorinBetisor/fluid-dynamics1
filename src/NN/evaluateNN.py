import numpy as np
import torch
import matplotlib.pyplot as plt
from neuralNetwork import FluidPDEAI

def evaluate_model(
    input_path="../data/inputs/input_0003.npy",
    target_path="../data/targets/target_0003.npy",
    model_path="NN/fluid_pde_ai.pt",
    in_channels=3,
    out_channels=3
):
    """
    Loads a trained model, runs prediction on input, compares with ground truth, and plots results.

    Args:
        input_path (str): Path to the input .npy file.
        target_path (str): Path to the target ground truth .npy file.
        model_path (str): Path to the saved trained model.
        in_channels (int): Number of input channels (e.g., 3 if using u, v, w only).
        out_channels (int): Number of output channels (usually 3).
    """
    # Load input
    x = np.load(input_path)
    x = x[:in_channels, :, :]  # Select only the needed channels
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, channels, H, W)

    # Load ground truth
    y_true = np.load(target_path)

    # Load model
    model = FluidPDEAI(in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred = model(x_tensor).squeeze(0).numpy()

    # Save prediction
    np.save("../data/predicted_output.npy", y_pred)
    print("✅ Prediction saved to ../data/predicted_output.npy")

    # Compute Mean Squared Error per channel
    mse_per_channel = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    titles = ["u (x-velocity)", "v (y-velocity)", "p (pressure)"]

    for i, mse in enumerate(mse_per_channel):
        print(f"MSE for {titles[i]}: {mse:.6f}")

    # Visualization
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        vmin = min(y_true[i].min(), y_pred[i].min())
        vmax = max(y_true[i].max(), y_pred[i].max())

        axs[i, 0].imshow(y_true[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f"Ground Truth {titles[i]}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(y_pred[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f"Prediction {titles[i]}")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(np.abs(y_true[i] - y_pred[i]), cmap="hot")
        axs[i, 2].set_title(f"Absolute Error {titles[i]}")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
