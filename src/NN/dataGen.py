import os
import numpy as np
import pandas as pd

# Where your ai_data CSVs are
AI_DATA_PATH = "ai_data"

# Where to save new input/target pairs
INPUT_SAVE_PATH = "data/inputs"
TARGET_SAVE_PATH = "data/targets"

# Make directories if needed
os.makedirs(INPUT_SAVE_PATH, exist_ok=True)
os.makedirs(TARGET_SAVE_PATH, exist_ok=True)

# Settings
dt_value = 0.005  # Your simulation dt (can be kept constant for now)
viscosity_value = 1.0 / 1000.0  # Inverse of Reynolds number if needed
output_interval = 10  # Same as in your C simulation
timesteps = sorted([int(f.split("_t")[1].split(".csv")[0]) for f in os.listdir(AI_DATA_PATH) if f.startswith("u_t")])

print(f"Found timesteps: {timesteps}")

def load_csv(name, t):
    return pd.read_csv(f"{AI_DATA_PATH}/{name}_t{t}.csv", header=None).values

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-5)

sample_idx = 0

for t in timesteps[:-1]:  # for t and t+Δt
    print(f"Processing timestep {t} -> {t + output_interval}")

    try:
        # Inputs at time t
        u_t = load_csv("u", t)
        v_t = load_csv("v", t)
        w_t = load_csv("w", t)
        obj_t = load_csv("obj", t)

        # Targets at time t+dt
        u_tp1 = load_csv("u", t + output_interval)
        v_tp1 = load_csv("v", t + output_interval)
        w_tp1 = load_csv("w", t + output_interval)

        # Normalize u, v, w
        u_t = normalize(u_t)
        v_t = normalize(v_t)
        w_t = normalize(w_t)

        u_tp1 = normalize(u_tp1)
        v_tp1 = normalize(v_tp1)
        w_tp1 = normalize(w_tp1)

        # Pressure placeholders (zeros)
        p_t = np.zeros_like(u_t)
        p_tp1 = np.zeros_like(u_tp1)

        # dt and viscosity layers
        dt_layer = np.ones_like(u_t) * dt_value
        viscosity_layer = np.ones_like(u_t) * viscosity_value

        # Assemble input (6 channels)
        input_tensor = np.stack([u_t, v_t, p_t, obj_t, dt_layer, viscosity_layer], axis=0)  # Shape: (6, H, W)

        # Assemble target (3 channels)
        target_tensor = np.stack([u_tp1, v_tp1, p_tp1], axis=0)  # Shape: (3, H, W)

        # Save
        np.save(f"{INPUT_SAVE_PATH}/input_{sample_idx:04d}.npy", input_tensor)
        np.save(f"{TARGET_SAVE_PATH}/target_{sample_idx:04d}.npy", target_tensor)

        sample_idx += 1

    except Exception as e:
        print(f"Skipping timestep {t} due to error: {e}")

print(f"✅ Done! Created {sample_idx} samples.")
