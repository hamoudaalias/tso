"""
Generate balanced gameplay dataset → train TSO LVQ1 → export.
Only keeps informative frames: near-misses + deaths.
"""
import json
import math
import random
import time

SCREEN_W = 600
SCREEN_H = 700
EMBED_DIM = 8
N_PROTOS = 8
N_GAMES = 2000
EPOCHS = 20

def encode(px, obstacles):
    v = [0.0] * EMBED_DIM
    v[0] = px / SCREEN_W
    if obstacles:
        nearest = min([o for o in obstacles if o['y'] < SCREEN_H - 30],
                      key=lambda o: o['y'], default=None)
        if nearest:
            v[1] = nearest['x'] / SCREEN_W
            v[2] = max(nearest['y'] / SCREEN_H, 0.0)
            v[3] = (nearest['x'] - px) / SCREEN_W
    v[4] = len(obstacles) / 10.0
    for i, o in enumerate(obstacles[:3]):
        v[5 + i] = min(max(o['y'] / SCREEN_H, 0.0), 1.0)
    return v

def cosine_sim(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-12
    nb = math.sqrt(sum(x*x for x in b)) + 1e-12
    return dot / (na * nb)

def simulate_game():
    px = SCREEN_W / 2
    obs = []
    alive, frame, spawn_timer = True, 0, 0
    speed = 3.0
    traj = []
    while alive and frame < 3000:
        frame += 1
        left = random.random() < 0.5
        state = encode(px, obs)
        px = max(min(px + (-5.0 if left else 5.0), SCREEN_W - 10.0), 10.0)
        speed = 3.0 + frame * 0.002
        si = max(int(40 - speed * 2), 8)
        spawn_timer += 1
        if spawn_timer >= si:
            spawn_timer = 0
            ox = px + random.uniform(-80, 80)
            ox = max(10, min(SCREEN_W - 10, ox))
            obs.append({'x': ox, 'y': -20})
        for o in obs:
            o['y'] += speed
        obs = [o for o in obs if o['y'] < SCREEN_H + 20]

        # Check death AND near-miss
        hit = False
        near_miss = False
        for o in obs:
            dx = abs(px - o['x'])
            dy = abs((SCREEN_H - 55) - o['y'])
            if dx < 21 and dy < 22:
                hit = True
            if dx < 40 and dy < 35:
                near_miss = True

        traj.append({'state': state, 'near_miss': near_miss, 'hit': hit, 'action': left})
        if hit:
            alive = False
    return traj

t0 = time.time()
print(f"Generating {N_GAMES} random games...")
survive_states = []
death_states = []

for i in range(N_GAMES):
    traj = simulate_game()
    for t in traj:
        if t['hit']:
            death_states.append((t['state'], 1))
        elif t['near_miss']:
            survive_states.append((t['state'], 0))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{N_GAMES} games — {len(survive_states)} near-miss, {len(death_states)} death")

# Balance: equal number of survive and death samples
n = min(len(survive_states), len(death_states))
n = min(n, 10000)
random.shuffle(survive_states)
random.shuffle(death_states)
all_states = survive_states[:n] + death_states[:n]
random.shuffle(all_states)

n_survive = sum(1 for _, c in all_states if c == 0)
n_death = sum(1 for _, c in all_states if c == 1)
print(f"Balanced dataset: {len(all_states)} samples ({n_survive} survive, {n_death} death)")

# Train LVQ1
print(f"\nTraining TSO LVQ1 ({EPOCHS} epochs, {N_PROTOS} proto/class)...")
prototypes = [[], []]
for _ in range(N_PROTOS):
    v = [random.uniform(-1, 1) for _ in range(EMBED_DIM)]
    n = math.sqrt(sum(x*x for x in v)) + 1e-12
    prototypes[0].append([x/n for x in v])
    v = [random.uniform(-1, 1) for _ in range(EMBED_DIM)]
    n = math.sqrt(sum(x*x for x in v)) + 1e-12
    prototypes[1].append([x/n for x in v])

lr = 0.05
for epoch in range(EPOCHS):
    random.shuffle(all_states)
    correct = 0
    for state, label in all_states:
        best_c, best_sim = 0, -1
        for c in (0, 1):
            for p in prototypes[c]:
                s = cosine_sim(state, p)
                if s > best_sim:
                    best_sim, best_c = s, c
        if best_c == label:
            correct += 1
        pi = max(range(len(prototypes[best_c])),
                key=lambda i: cosine_sim(state, prototypes[best_c][i]))
        if best_c == label:
            for j in range(EMBED_DIM):
                prototypes[best_c][pi][j] += lr * (state[j] - prototypes[best_c][pi][j])
        else:
            for j in range(EMBED_DIM):
                prototypes[best_c][pi][j] -= lr * (state[j] - prototypes[best_c][pi][j])
            if prototypes[label]:
                ci = max(range(len(prototypes[label])),
                        key=lambda i: cosine_sim(state, prototypes[label][i]))
                for j in range(EMBED_DIM):
                    prototypes[label][ci][j] += lr * (state[j] - prototypes[label][ci][j])
        n = math.sqrt(sum(x*x for x in prototypes[best_c][pi])) + 1e-12
        prototypes[best_c][pi] = [x/n for x in prototypes[best_c][pi]]
    acc = correct / len(all_states) * 100
    print(f"  Epoch {epoch:>2}: acc = {acc:.1f}%")

dt = time.time() - t0
print(f"\n✅ Training complete in {dt:.1f}s — accuracy: {acc:.1f}%")

agent_data = {"prototypes": prototypes}
stats_data = {
    "n_games": N_GAMES,
    "n_samples": len(all_states),
    "n_survive": n_survive,
    "n_death": n_death,
    "n_protos": N_PROTOS,
    "epochs": EPOCHS,
    "accuracy": round(acc, 1),
    "training_time": round(dt, 1)
}

with open("best_agent.json", "w") as f:
    json.dump(agent_data, f)
with open("stats.json", "w") as f:
    json.dump(stats_data, f)
print(f"   Exported best_agent.json + stats.json")
