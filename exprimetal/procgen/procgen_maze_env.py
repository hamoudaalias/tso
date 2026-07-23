"""
ProcgenHeistEnv v2 — T-maze procédural.

Règles strictes :
- Exit et gem randomisés à chaque reset
- Observation 5×5 direction-dépendante (MiniGrid-style)
- Aucun accès à self.grid depuis l'agent
- Aucune coordonnée absolue dans l'observation
- Victoire uniquement si gem COLLECTED + agent à l'exit
"""
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class ProcgenHeistEnv(gym.Env):
    """T-maze 11×11. Corridor horizontal y=5 (x=1..9), vertical x=5 (y=1..9).
    Start (5,5). Exit TOP=(5,2) ou BOTTOM=(5,8). Gem LEFT=(1,5) ou RIGHT=(9,5).

    Observation 5×5 int32, orientée vers le haut (dir-dependent) :
      0=vide 1=mur 2=agent 3=gemme 5=sortie
    Action: 0=turn_left, 1=turn_right, 2=forward, 3=stay
    """
    metadata = {"render_modes": []}

    def __init__(self, max_steps=150):
        super().__init__()
        self.N = 11
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=5, shape=(5, 5), dtype=np.int32)

    def _generate(self):
        self.grid = np.ones((self.N, self.N), dtype=np.int32)
        for x in range(1, self.N - 1):
            self.grid[5, x] = 0  # corridor horizontal
        for y in range(1, self.N - 1):
            self.grid[y, 5] = 0  # corridor vertical

        self.exit_pos = random.choice([(5, 2), (5, 8)])
        self.gem_pos = random.choice([(1, 5), (9, 5)])
        self.agent_pos = [5, 5]
        self.agent_dir = random.choice([0, 1, 2, 3])
        self.gem_collected = False

    def _observe(self):
        ox, oy = self.agent_pos
        view = np.zeros((5, 5), dtype=np.int32)

        for r in range(5):
            for c in range(5):
                if self.agent_dir == 0:    dx, dy = c - 2, r - 2
                elif self.agent_dir == 1:   dx, dy = 2 - r, c - 2
                elif self.agent_dir == 2:   dx, dy = 2 - c, 2 - r
                else:                       dx, dy = r - 2, 2 - c

                gx, gy = ox + dx, oy + dy

                if 0 <= gx < self.N and 0 <= gy < self.N:
                    if (gx, gy) == self.exit_pos:
                        view[r, c] = 5
                    elif not self.gem_collected and (gx, gy) == self.gem_pos:
                        view[r, c] = 3
                    elif int(self.grid[gy, gx]) == 1:
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
        self.gem_collected = False
        return self._observe(), {}

    def step(self, action):
        self.step_count += 1
        a_dir = self.agent_dir

        if action == 0: self.agent_dir = (self.agent_dir + 3) % 4
        elif action == 1: self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dx, dy = DIRS[a_dir]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.N and 0 <= ny < self.N and self.grid[ny, nx] != 1:
                self.agent_pos = [nx, ny]

        if not self.gem_collected and tuple(self.agent_pos) == self.gem_pos:
            self.gem_collected = True

        obs = self._observe()
        won = self.gem_collected and tuple(self.agent_pos) == self.exit_pos
        return obs, float(won), won, self.step_count >= self.max_steps, {}
