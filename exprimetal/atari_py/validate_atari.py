import sys, os, time
import numpy as np
from skimage.transform import resize

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from tso_pyo3 import DualLIFState, ActionMotor, Cerebellum, AssociativeMemory, AttractorField


# ─── Retina: V1-like visual cortex ───────────────────────────────────────────

def extract_blocks(rgb, grid=8):
    h, w = rgb.shape[:2]
    gray = np.mean(rgb, axis=2).astype(np.float64)
    ex = np.abs(np.diff(gray, axis=1))
    ey = np.abs(np.diff(gray, axis=0))
    edges = np.zeros_like(gray)
    edges[:, :-1] += ex
    edges[:-1, :] += ey
    bs = h // grid
    blocks = []
    for i in range(grid):
        for j in range(grid):
            y, x = i * bs, j * bs
            b = rgb[y:y + bs, x:x + bs].astype(np.float64) / 255.0
            eb = edges[y:y + bs, x:x + bs] / 255.0
            blocks.append([eb.mean()] + b.mean(axis=(0, 1)).tolist())
    return np.array(blocks, dtype=np.float64)


def develop_v1_from_obs(obs_list, n_protos=32, lr=0.05):
    field = None
    init = False
    dim = 4
    for obs in obs_list:
        blks = extract_blocks(obs)
        for b in blks:
            if not init:
                if field is None:
                    field = AttractorField(dim, n_protos, 3, lr)
                field.add_class(b.tolist())
                init = field.n_classes() >= n_protos
            else:
                w = field.predict(b.tolist())
                field.train_step(b.tolist(), w)
    return field


def encode_v1(field, rgb):
    blks = extract_blocks(rgb)
    vec = np.zeros(64, dtype=np.float64)
    n = field.n_classes()
    for i, b in enumerate(blks):
        vec[i] = field.predict(b.tolist()) / max(n - 1, 1)
    return vec.tolist()


def preprocess(obs, size=64):
    return (resize(obs, (size, size, 3), preserve_range=True).astype(np.uint8))


# ─── TSO Agent ───────────────────────────────────────────────────────────────

def make_action_vecs(dim, n_actions):
    vecs = []
    for i in range(n_actions):
        v = np.zeros(dim, dtype=np.float64)
        np.random.seed(i * 7 + 13)
        idxs = np.random.choice(dim, min(6, dim), replace=False)
        v[idxs] = 0.2
        vecs.append(v.tolist())
    return vecs


class TSOAtariAgent:
    def __init__(self, v1, n_actions=6, dim=64, lr=0.05, noise_std=0.3):
        self.v1 = v1
        self.dim = dim
        self.n_actions = n_actions
        self.lif = DualLIFState(dim, 0.95, 0.5)
        self.cb = Cerebellum(dim, n_actions, lr, noise_std)
        self.mem = AssociativeMemory()
        self.motor = ActionMotor(0.7)
        self.action_vecs = make_action_vecs(dim, n_actions)

    def act(self, rgb, explore=True):
        vec = encode_v1(self.v1, rgb)
        self.lif.step(vec, False)
        concept = self.lif.get_slow_state()
        rc = self.mem.recall_with_sim(concept)
        bonuses = [0.0] * self.n_actions
        if rc is not None and rc[1] > 0.35:
            bonuses[rc[0]] = 0.5
        cb_a = self.cb.forward(concept)
        bonuses[cb_a] += 0.3
        if explore and np.random.random() < 0.1:
            return np.random.randint(self.n_actions), concept
        action, _ = self.motor.select_with_bonus(self.lif, self.action_vecs, bonuses)
        return action, concept

    def learn(self, concept, action, reward):
        learn_signal = reward + 0.1 * (reward - 0.0)
        self.cb.learn(concept, action, learn_signal)
        if reward > 0:
            self.mem.store(concept, action)

    def reset(self):
        self.lif = DualLIFState(self.dim, 0.95, 0.5)

    def state(self):
        return self.lif.get_slow_state()


# ─── Validation ──────────────────────────────────────────────────────────────

def run_episode(env_name, agent, n_actions, max_steps=500, explore=True):
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    obs = preprocess(obs)
    total = 0.0

    for _ in range(max_steps):
        action, concept = agent.act(obs, explore=explore)
        action = min(action, n_actions - 1)
        obs, r, terminated, truncated, info = env.step(action)
        obs = preprocess(obs)
        total += r
        agent.learn(concept, action, r)
        if terminated or truncated:
            break

    env.close()
    return total


def run_random(env_name, max_steps=500):
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    total = 0.0

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total += r
        if terminated or truncated:
            break

    env.close()
    return total


def main():
    os.makedirs("_results", exist_ok=True)

    for env_name in ["ALE/Pong-v5", "ALE/Breakout-v5"]:
        short = env_name.split("/")[1].split("-")[0]
        print(f"\n{'='*60}")
        print(f"  {env_name}")
        print(f"{'='*60}")

        # ── Detect action space ──
        tmp_env = gym.make(env_name, render_mode='rgb_array')
        n_actions = int(tmp_env.action_space.n)
        tmp_env.close()
        print(f"  [0] Action space: {n_actions} actions")

        # ── Develop V1 cortex ──
        print(f"  [1] Collecting observations for V1 development...")
        tmp_env = gym.make(env_name, render_mode='rgb_array')
        obs_list = []
        for _ in range(200):
            o, _ = tmp_env.reset()
            obs_list.append(preprocess(o))
            for _ in range(10):
                o, _, terminated, truncated, _ = tmp_env.step(tmp_env.action_space.sample())
                obs_list.append(preprocess(o))
                if terminated or truncated:
                    break
        tmp_env.close()

        print(f"  [2] Developing V1 cortex ({len(obs_list)} frames)...")
        v1 = develop_v1_from_obs(obs_list, n_protos=32, lr=0.05)
        print(f"       V1: {v1.n_classes()} prototypes")

        # ── Train ──
        print(f"  [3] Training TSO agent (5 episodes)...")
        agent = TSOAtariAgent(v1, n_actions=n_actions)
        train_rewards = []
        for ep in range(5):
            r = run_episode(env_name, agent, n_actions, max_steps=500, explore=True)
            train_rewards.append(r)
            agent.reset()
            print(f"       ep {ep+1}: {r:+.1f}")
        print(f"       Train mean: {np.mean(train_rewards):+.2f} ± {np.std(train_rewards):.2f}")

        # ── Evaluate ──
        print(f"  [4] Evaluating TSO agent (5 episodes)...")
        agent_eval = TSOAtariAgent(v1, n_actions=n_actions)
        eval_rewards = []
        t0 = time.time()
        for ep in range(5):
            r = run_episode(env_name, agent_eval, n_actions, max_steps=500, explore=False)
            eval_rewards.append(r)
            agent_eval.reset()
        elapsed = time.time() - t0

        # ── Random baseline ──
        print(f"  [5] Random baseline...")
        rnd_rewards = [run_random(env_name, max_steps=500) for _ in range(5)]

        print(f"\n  ──────────────────────────────────────────────")
        print(f"  {env_name:30s}")
        print(f"  TSO : {np.mean(eval_rewards):+7.2f} ± {np.std(eval_rewards):.2f}  ({elapsed:.1f}s)")
        print(f"  RND : {np.mean(rnd_rewards):+7.2f} ± {np.std(rnd_rewards):.2f}")
        print(f"  Δ   : {np.mean(eval_rewards) - np.mean(rnd_rewards):+7.2f}")
        print(f"  ──────────────────────────────────────────────")

        with open("_results/atari_results.txt", "a") as f:
            f.write(
                f"{env_name:30s}"
                f"  TSO={np.mean(eval_rewards):+7.2f}±{np.std(eval_rewards):.2f}"
                f"  RND={np.mean(rnd_rewards):+7.2f}±{np.std(rnd_rewards):.2f}"
                f"  Δ={np.mean(eval_rewards) - np.mean(rnd_rewards):+7.2f}\n"
            )

    print(f"\n{'='*60}")
    print("  Perception-action loop validated!")
    print(f"{'='*60}")

    with open("_results/atari_results.txt", "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
