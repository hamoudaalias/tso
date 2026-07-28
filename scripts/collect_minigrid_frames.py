#!/usr/bin/env python3
"""Collect 5000 real Minigrid frames, save as .npy for VAE training."""

import numpy as np
from minigrid.envs import EmptyEnv
from tqdm import tqdm

N_FRAMES = 5000
ENV_SIZE = 8  # EmptyEnv-8×8 → obs image shape (7, 7, 3)
OUT_PATH = "scripts/minigrid_frames_5000.npy"

env = EmptyEnv(size=ENV_SIZE)
frames = np.zeros((N_FRAMES, 7, 7, 3), dtype=np.float32)

obs, _ = env.reset()
for i in tqdm(range(N_FRAMES), desc="Collecting frames"):
    frames[i] = obs["image"].astype(np.float32) / 255.0
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

np.save(OUT_PATH, frames)
print(f"Saved {N_FRAMES} frames → {OUT_PATH}  (shape: {frames.shape})")
