#!/usr/bin/env python3
"""Collect GridWorld frames as 25D one-hot + train VAE, export .bin.

GridWorld 5×5: 25 cells, each encoded as:
  0.0 = empty   0.5 = wall   1.0 = water
  agent bonus: add 2.0 to current cell (so cell with agent+water = 3.0)
"""

import numpy as np
import struct, random

W, H = 5, 5
N_FRAMES = 10000
WATER_CELLS = [(1,1), (3,3), (1,4)]
OUT_PATH = "scripts/gridworld_data_10k.bin"

def render_grid(ax, ay, water_cells):
    """Render 5×5 grid as 25D array."""
    grid = np.zeros(25, dtype=np.float64)
    # walls on edges
    for x in range(W):
        for y in range(H):
            if x == 0 or x == W-1 or y == 0 or y == H-1:
                grid[y * W + x] = 0.5  # wall
    # water
    for wx, wy in water_cells:
        grid[wy * W + wx] = 1.0  # water
    # agent (overwrites, not additive)
    grid[ay * W + ax] = 2.0  # agent flag, distinct
    return grid

# Collect frames via random walk
frames = np.zeros((N_FRAMES, 25), dtype=np.float64)
x, y = 2, 2
for i in range(N_FRAMES):
    frames[i] = render_grid(x, y, WATER_CELLS)
    # random step
    dx, dy = random.choice([(0,-1),(0,1),(-1,0),(1,0)])
    nx, ny = x + dx, y + dy
    if 0 <= nx < W and 0 <= ny < H:
        # don't step into walls (edges)
        if nx > 0 and nx < W-1 and ny > 0 and ny < H-1:
            x, y = nx, ny
        else:
            # bounce
            pass

print(f"Collected {N_FRAMES} GridWorld frames (25D each)")

with open(OUT_PATH, 'wb') as f:
    f.write(struct.pack('<II', N_FRAMES, 25))
    f.write(frames.tobytes())
print(f"Saved → {OUT_PATH} ({N_FRAMES * 25 * 8 / 1024:.0f} KB)")
