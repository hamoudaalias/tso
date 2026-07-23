"""
Generate gameplay dataset → store ALL states as memory (no prototypes).
TSO decides by cosine similarity to past good/bad states.
"""
import json
import math
import random
import time

SCREEN_W = 600
SCREEN_H = 700
EMBED_DIM = 8
N_GAMES = 5000

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
    frames = []
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

        near_miss = any(abs(px - o['x']) < 40 and abs((SCREEN_H - 55) - o['y']) < 35 for o in obs)
        hit = any(abs(px - o['x']) < 21 and abs((SCREEN_H - 55) - o['y']) < 22 for o in obs)

        frames.append({'state': state, 'action': left, 'near_miss': near_miss, 'hit': hit})
        if hit:
            alive = False
    return frames

t0 = time.time()
print(f"Generating {N_GAMES} random games...")

memory = {'good': [], 'bad': []}

for i in range(N_GAMES):
    traj = simulate_game()
    for t in traj:
        if t['hit']:
            memory['bad'].append({'state': t['state'], 'action': t['action']})
        elif t['near_miss']:
            memory['good'].append({'state': t['state'], 'action': t['action']})
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{N_GAMES} — memory: {len(memory['good'])} good + {len(memory['bad'])} bad")

# Balance: equal numbers
n = min(len(memory['good']), len(memory['bad']))
n = min(n, 50000)
random.shuffle(memory['good'])
random.shuffle(memory['bad'])
memory['good'] = memory['good'][:n]
memory['bad'] = memory['bad'][:n]

print(f"Memory: {len(memory['good'])} good + {len(memory['bad'])} bad")

# Validate on a held-out set
print("\nValidating...")
correct = 0
total = 0
test_set = [(s['state'], True) for s in memory['good'][:1000]] + [(s['state'], False) for s in memory['bad'][:1000]]
for state_vec, expected_good in test_set:
    best_sim_good = max([cosine_sim(state_vec, m['state']) for m in memory['good']], default=-1)
    best_sim_bad = max([cosine_sim(state_vec, m['state']) for m in memory['bad']], default=-1)
    pred_good = best_sim_good > best_sim_bad
    if pred_good == expected_good:
        correct += 1
    total += 1
print(f"  Accuracy: {correct/total*100:.1f}% ({correct}/{total})")

# Export
data = {
    'good_states': [m['state'] for m in memory['good']],
    'good_actions': [m['action'] for m in memory['good']],
    'bad_states': [m['state'] for m in memory['bad']],
    'bad_actions': [m['action'] for m in memory['bad']],
}
with open("memory.json", "w") as f:
    json.dump(data, f)

stats = {
    "n_games": N_GAMES,
    "n_good": len(memory['good']),
    "n_bad": len(memory['bad']),
    "training_time": round(time.time() - t0, 1),
    "accuracy": round(correct/total*100, 1),
    "method": "cosine_memory_knn"
}
with open("stats.json", "w") as f:
    json.dump(stats, f)

dt = time.time() - t0
print(f"\n✅ Done in {dt:.1f}s — accuracy: {correct/total*100:.1f}%")
print(f"   Exported memory.json + stats.json")
