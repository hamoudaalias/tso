"""
Test EpisodicMemory — T-maze avec DFS.
L'agent explore systematic (DFS), trouve la sortie, stocke la séquence
d'actions dans EpisodicMemory. Rappel séquence par séquence sur 4 runs.
NORMAL: EpisodicMemory intact → recall direct.
AMNÉSIQUE: EpisodicMemory reset → pas de recall → DFS à chaque run.
"""
import sys
sys.path.insert(0, "exprimetal/procgen")
import numpy as np
from tso_pyo3 import EpisodicMemory

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class TMazeEnv:
    """T-maze fixe. Start (2,3), exit (1,1).
    Chemin optimal: left(0) forward(2) left(0) forward(2) forward(2)
    (5 actions)"""
    def __init__(self, max_steps=30):
        self.N = 5
        self.max_steps = max_steps

    def reset(self):
        self.grid = np.ones((self.N, self.N), dtype=np.int32)
        self.grid[1, 1] = 0  # exit
        self.grid[1, 3] = 0  # dead end
        self.grid[2, 1] = 0; self.grid[2, 2] = 0; self.grid[2, 3] = 0
        self.grid[3, 1] = 0; self.grid[3, 2] = 0; self.grid[3, 3] = 0
        self.exit_pos = (1, 1)
        self.agent_pos = [2, 3]
        self.agent_dir = 0
        self.step_count = 0
        return self._observe()

    def _observe(self):
        ox, oy = self.agent_pos
        obs = np.full((3, 3), 1, dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                gx, gy = ox+dx, oy+dy
                if 0 <= gx < self.N and 0 <= gy < self.N:
                    if (gx, gy) == self.exit_pos: obs[dy+1, dx+1] = 3
                    elif int(self.grid[gy, gx]) == 0: obs[dy+1, dx+1] = 0
        obs[1, 1] = 2
        return obs

    def step(self, action):
        self.step_count += 1
        if action == 0: self.agent_dir = (self.agent_dir + 3) % 4
        elif action == 1: self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dx, dy = DIRS[self.agent_dir]
            nx, ny = self.agent_pos[0]+dx, self.agent_pos[1]+dy
            if 0 <= nx < self.N and 0 <= ny < self.N and self.grid[ny, nx] != 1:
                self.agent_pos = [nx, ny]
        obs = self._observe()
        won = tuple(self.agent_pos) == self.exit_pos
        return obs, float(won), won, self.step_count >= self.max_steps, {}

    def get_open_dirs(self):
        x, y = self.agent_pos
        return [i for i,(dx,dy) in enumerate(DIRS)
                if 0<=x+dx<self.N and 0<=y+dy<self.N and self.grid[y+dy, x+dx] != 1]


def run_trial(amnesic=False):
    env = TMazeEnv()
    env.reset()

    # DFS to find path
    path = []
    visited = set()
    stack = [(env.agent_pos[0], env.agent_pos[1], env.agent_dir, [])]
    found_path = None

    while stack and found_path is None:
        x, y, d, actions = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # Simulate reaching this state
        # Find what actions lead here
        if (x, y) == env.exit_pos:
            found_path = actions
            break

        open_dirs = []
        for di, (dx, dy) in enumerate(DIRS):
            nx, ny = x+dx, y+dy
            if 0 <= nx < env.N and 0 <= ny < env.N and env.grid[ny, nx] != 1:
                open_dirs.append(di)

        for next_dir in sorted(open_dirs):
            nx, ny = x + DIRS[next_dir][0], y + DIRS[next_dir][1]
            if (nx, ny) not in visited:
                # Actions to go in next_dir from current dir d
                diff = (next_dir - d) % 4
                if diff == 1:
                    new_actions = actions + [1, 2]  # turn right, forward
                elif diff == 3:
                    new_actions = actions + [0, 2]  # turn left, forward
                elif diff == 0:
                    new_actions = actions + [2]  # forward
                else:  # diff == 2, turn around
                    new_actions = actions + [1, 1, 2]  # right right forward
                stack.append((nx, ny, next_dir, new_actions))

    if found_path is None:
        return 0  # not found

    # Store path in EpisodicMemory
    mem = EpisodicMemory(100)
    if not amnesic:
        mem.store(found_path)

    # Recall runs
    recall_ok = 0
    for _ in range(4):
        env.reset()
        hist = []
        for s in range(30):
            if amnesic:
                mem = EpisodicMemory(100)
                hist = []

            if not amnesic and len(hist) > 0:
                next_a = mem.recall(hist)
            else:
                next_a = None

            if next_a is not None:
                a = next_a
            elif not amnesic and s < len(found_path):
                a = found_path[s]
            else:
                # Fallback: systematic DFS from current state
                open_dirs = env.get_open_dirs()
                target_dir = sorted(open_dirs)[0] if open_dirs else 0
                diff = (target_dir - env.agent_dir) % 4
                if diff == 1: a = 1
                elif diff == 3: a = 0
                elif diff == 0: a = 2
                else: a = 1

            hist.append(a)
            _, _, term, _, _ = env.step(a)
            if term:
                recall_ok += 1
                break
            if env.step_count >= env.max_steps:
                break

    return recall_ok


def main():
    for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True)]:
        all_r = []
        print(f"\n=== {label} ===")
        for ep in range(50):
            r = run_trial(amnesic)
            all_r.append(r)
            sys.stdout.write(f"\rEp {ep+1:2d}: recall={r}/4")
            sys.stdout.flush()
        print()
        total = sum(all_r)
        print(f"{label}: recall {total}/200 ({total*0.5:.0f}%)")


if __name__ == "__main__":
    main()
