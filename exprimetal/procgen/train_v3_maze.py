"""
TSO v3 — Procgen Maze (version rapide).
Démontre :
1. DualLIF → état caché du labyrinthe par intégration temporelle
2. Cerebellum → apprentissage moteur Hebbien
3. AssociativeMemory → rappel d'actions récompensées

NORMAL vs AMNÉSIQUE vs ALEA.
"""
import sys, numpy as np, gym, procgen
sys.path.insert(0, "exprimetal/procgen")
from tso_pyo3 import DualLIFState, ActionMotor, AssociativeMemory, Cerebellum
from retina_v3 import develop_v1, encode_v1


def train_v1(env_name="maze", n_steps=200, n_protos=16):
    print(f"Développement V1...")
    env = gym.make(f"procgen-{env_name}-v0", start_level=0, num_levels=50)
    field = develop_v1(env, n_steps=n_steps, n_protos=n_protos)
    env.close()
    print(f"  {field.n_classes()} protos")
    return field


def make_action_vecs(dim):
    vecs = []
    for i in range(15):
        v = np.zeros(dim, dtype=np.float64)
        np.random.seed(i * 7 + 13)
        idxs = np.random.choice(dim, min(6, dim), replace=False)
        v[idxs] = 0.2
        vecs.append(v.tolist())
    return vecs


def run_episode(env, v1, cb=None, mem=None, train=True, amnesic=False, max_steps=250):
    dim = 64
    lif = DualLIFState(dim, 0.95, 0.5)
    motor = ActionMotor(0.7)
    av = make_action_vecs(dim)

    if amnesic or cb is None:
        cb = Cerebellum(dim, 15, 0.05, 0.3)
    if amnesic or mem is None:
        mem = AssociativeMemory()

    obs = env.reset()
    total = 0.0

    for step in range(max_steps):
        vec = encode_v1(v1, obs)
        lif.step(vec, False)
        concept = lif.get_slow_state()

        rc = mem.recall_with_sim(concept)
        bonuses = [0.0] * 15
        if rc is not None and rc[1] > 0.35:
            bonuses[rc[0]] = 0.5

        cb_a = cb.forward(concept)
        bonuses[cb_a] += 0.3 if train else 0.5

        if train and np.random.random() < 0.1:
            action = np.random.randint(15)
        else:
            action, _ = motor.select_with_bonus(lif, av, bonuses)

        next_obs, reward, done, info = env.step(action)
        total += reward

        if train:
            nv = encode_v1(v1, next_obs)
            lif.step(nv, False)
            n_concept = lif.get_slow_state()
            learn_signal = reward + 0.1 * (reward - 0.0)
            cb.learn(concept, action, learn_signal)
            if reward > 0:
                mem.store(concept, action)

        obs = next_obs
        if done:
            break

    return total > 0, total, step + 1


def run_random(env, max_steps=250):
    obs = env.reset()
    total = 0.0
    for step in range(max_steps):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        total += r
        if done:
            break
    return total > 0, total, step + 1


def main():
    n_train = 20
    n_eval = 20

    v1 = train_v1("maze", n_steps=200, n_protos=16)

    print(f"\n=== Entraînement ({n_train} eps) ===")
    env = gym.make("procgen-maze-v0", start_level=0, num_levels=100)
    cb = Cerebellum(64, 15, 0.05, 0.3)
    mem = AssociativeMemory()

    for ep in range(n_train):
        ok, r, s = run_episode(env, v1, cb, mem, train=True, amnesic=False)
        sys.stdout.write(f"\r  Ep {ep+1:2d}: r={r:+.3f} {'WIN' if ok else 'LOSS':5s}")
        sys.stdout.flush()
    env.close()
    print()

    print(f"\n=== Évaluation ({n_eval} eps) ===")
    te = gym.make("procgen-maze-v0", start_level=100, num_levels=50)
    wn, rn = 0, []
    for ep in range(n_eval):
        ok, r, s = run_episode(te, v1, cb, mem, train=False, amnesic=False)
        rn.append(r)
        if ok: wn += 1
        sys.stdout.write(f"\r  NORMAL Ep {ep+1:2d}: r={r:+.3f} {'WIN' if ok else 'LOSS':5s}")
        sys.stdout.flush()
    te.close()
    print()

    te2 = gym.make("procgen-maze-v0", start_level=100, num_levels=50)
    wa, ra = 0, []
    for ep in range(n_eval):
        ok, r, s = run_episode(te2, v1, train=False, amnesic=True)
        ra.append(r)
        if ok: wa += 1
        sys.stdout.write(f"\r  AMNÉSIQUE Ep {ep+1:2d}: r={r:+.3f} {'WIN' if ok else 'LOSS':5s}")
        sys.stdout.flush()
    te2.close()
    print()

    te3 = gym.make("procgen-maze-v0", start_level=100, num_levels=50)
    wr, rr = 0, []
    for ep in range(n_eval):
        ok, r, s = run_random(te3)
        rr.append(r)
        if ok: wr += 1
    te3.close()

    print(f"\n=== Résultats procgen-maze ===")
    print(f"{'Agent':<15} {'Wins':>8} {'Reward μ':>10}")
    print("-" * 35)
    print(f"{'NORMAL':<15} {wn:>4}/{n_eval}  {np.mean(rn):>+8.3f}")
    print(f"{'AMNÉSIQUE':<15} {wa:>4}/{n_eval}  {np.mean(ra):>+8.3f}")
    print(f"{'ALÉATOIRE':<15} {wr:>4}/{n_eval}  {np.mean(rr):>+8.3f}")


if __name__ == "__main__":
    main()
