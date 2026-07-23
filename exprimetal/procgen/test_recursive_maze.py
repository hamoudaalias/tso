"""
Test ProcgenMazeRecursive — ablation EpisodicMemory.
Même maze, même chemin. Stocké dans EpisodicMemory.
Rappel PUR : aucune connaissance du chemin optimal.
NORMAL peut rappeler depuis EpisodicMemory. AMNÉSIQUE ne peut pas.
"""
import sys, random
sys.path.insert(0, "exprimetal/procgen")
from tso_pyo3 import EpisodicMemory
from procgen_recursive_maze import ProcgenMazeRecursiveEnv

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def bfs_optimal_path(grid, start, exit_pos):
    N = grid.shape[0]
    q = [(start[0], start[1], [])]
    visited = set()
    while q:
        x, y, path = q.pop(0)
        if (x, y) == exit_pos:
            return path
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for di, (dx, dy) in enumerate(DIRS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and grid[ny, nx] != 1:
                q.append((nx, ny, path + [di]))
    return None


def path_to_actions(path, start_dir=0):
    actions = []
    d = start_dir
    for next_dir in path:
        diff = (next_dir - d) % 4
        if diff == 0:
            actions.append(2)
        elif diff == 1:
            actions.append(1)
            actions.append(2)
        elif diff == 3:
            actions.append(0)
            actions.append(2)
        else:
            actions.append(1)
            actions.append(1)
            actions.append(2)
        d = next_dir
    return actions


def run_trial(env, seed, amnesic=False, max_steps=500):
    random.seed(seed)
    obs, _ = env.reset(seed=seed)
    grid = env.grid.copy()
    agent_pos = list(env.agent_pos)
    agent_dir = env.agent_dir
    exit_pos = env.exit_pos
    start_dir = agent_dir

    opt_path = bfs_optimal_path(grid, tuple(agent_pos), exit_pos)
    if opt_path is None:
        return 0, 0

    actions = path_to_actions(opt_path, start_dir)

    mem = EpisodicMemory(len(actions) + 10)
    if not amnesic:
        mem.store(actions)

    recall_ok = 0
    n_recall = 4
    for _ in range(n_recall):
        env.grid = grid.copy()
        env.agent_pos = list(agent_pos)
        env.agent_dir = agent_dir
        env.exit_pos = exit_pos
        env.step_count = 0
        obs = env._observe()

        hist = []
        for step in range(max_steps):
            if amnesic:
                mem = EpisodicMemory(len(actions) + 10)

            if not amnesic and len(hist) > 0:
                next_a = mem.recall(hist)
            else:
                next_a = None

            if next_a is not None:
                action = next_a
            else:
                action = random.choice([0, 1, 2])

            hist.append(action)
            if not amnesic and len(hist) > 20:
                hist = hist[-20:]

            obs, reward, done, trunc, _ = env.step(action)
            if done:
                recall_ok += 1
                break
            if trunc:
                break

    return recall_ok, n_recall


def main():
    n_mazes = 50
    for label, amnesic in [("NORMAL", False), ("AMNÉSIQUE", True)]:
        total_recall = 0
        total_possible = 0
        print(f"\n=== {label} ===")
        for maze_id in range(n_mazes):
            env = ProcgenMazeRecursiveEnv(n=11, max_steps=500)
            recall, possible = run_trial(env, seed=maze_id + 100, amnesic=amnesic)
            total_recall += recall
            total_possible += possible
            sys.stdout.write(f"\rMaze {maze_id+1:2d}: recall {recall}/{possible}")
            sys.stdout.flush()
            env.close()
        print()
        pct = total_recall / total_possible * 100
        print(f"{label}: {total_recall}/{total_possible} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
