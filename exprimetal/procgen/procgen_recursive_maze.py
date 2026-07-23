"""
ProcgenMazeRecursive — Labyrinthe procédural généré par recursive backtracking.
Vue partielle 5×5 direction-dépendante (MiniGrid-style).
L'agent doit explorer les embranchements et se souvenir du chemin.

Ceci corrige le biais identifié dans l'audit (paper.md) :
- Génération procédurale réelle (pas de layout fixe)
- Vision partielle uniquement (pas d'accès à la grille globale)
- Mémoire épisodique pour rappel de chemin
"""
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def generate_maze(n):
    grid = np.ones((n, n), dtype=np.int32)

    def carve(y, x):
        grid[y, x] = 0
        order = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(order)
        for dy, dx in order:
            ny, nx = y + dy, x + dx
            my, mx = y + dy // 2, x + dx // 2
            if 0 <= ny < n and 0 <= nx < n and grid[ny, nx] == 1:
                grid[my, mx] = 0
                carve(ny, nx)

    carve(1, 1)
    grid[1, 1] = 0
    grid[n - 2, n - 2] = 0
    return grid


class ProcgenMazeRecursiveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n=11, max_steps=200):
        super().__init__()
        self.N = n if n % 2 == 1 else n + 1
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=5, shape=(5, 5), dtype=np.int32)

    def _generate(self):
        self.grid = generate_maze(self.N)
        corners = []
        for y in range(1, self.N - 1):
            for x in range(1, self.N - 1):
                if self.grid[y, x] == 0:
                    dist = abs(y - 1) + abs(x - 1)
                    if dist > self.N * 0.4:
                        corners.append((x, y))
        if not corners:
            corners = [(self.N - 3, self.N - 3)]

        self.exit_pos = random.choice(corners)
        self.agent_pos = [1, 1]
        self.agent_dir = 0

    def _observe(self):
        ox, oy = self.agent_pos
        view = np.zeros((5, 5), dtype=np.int32)

        for r in range(5):
            for c in range(5):
                if self.agent_dir == 0:
                    dx, dy = c - 2, r - 2
                elif self.agent_dir == 1:
                    dx, dy = 2 - r, c - 2
                elif self.agent_dir == 2:
                    dx, dy = 2 - c, 2 - r
                else:
                    dx, dy = r - 2, 2 - c

                gx, gy = ox + dx, oy + dy

                if 0 <= gx < self.N and 0 <= gy < self.N:
                    if (gx, gy) == self.exit_pos:
                        view[r, c] = 5
                    elif self.grid[gy, gx] == 1:
                        view[r, c] = 1
                else:
                    view[r, c] = 1

        view[2, 2] = 2
        return view

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._generate()
        self.step_count = 0
        return self._observe(), {}

    def step(self, action):
        self.step_count += 1
        a_dir = self.agent_dir

        if action == 0:
            self.agent_dir = (self.agent_dir + 3) % 4
        elif action == 1:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dx, dy = DIRS[a_dir]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.N and 0 <= ny < self.N and self.grid[ny, nx] != 1:
                self.agent_pos = [nx, ny]

        obs = self._observe()
        won = tuple(self.agent_pos) == self.exit_pos
        return obs, float(won), won, self.step_count >= self.max_steps, {}
