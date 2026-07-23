"""
TSO Snail Jumper — memory-based (cosine k-NN) + tkinter.
Loads pre-trained memory, shows intro, plays on GO.
"""
import json
import math
import random
import tkinter as tk

SCREEN_W = 600
SCREEN_H = 700
PLAYER_Y = SCREEN_H - 100
EMBED_DIM = 8

def cosine_sim(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-12
    nb = math.sqrt(sum(x*x for x in b)) + 1e-12
    return dot / (na * nb)

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

# ── Load memory ───────────────────────────────────────────────
with open("memory.json") as f:
    mem = json.load(f)
good_states = mem['good_states']
good_actions = mem['good_actions']
bad_states = mem['bad_states']
bad_actions = mem['bad_actions']

stats = {"n_games": 0, "n_good": 0, "n_bad": 0, "training_time": 0, "accuracy": 0}
try:
    with open("stats.json") as f:
        stats = json.load(f)
except FileNotFoundError:
    pass

def decide(state):
    """Find nearest neighbor in memory → act like they did."""
    best_sim_good = max([cosine_sim(state, s) for s in good_states], default=-1)
    best_sim_bad = max([cosine_sim(state, s) for s in bad_states], default=-1)
    if best_sim_good > best_sim_bad:
        # Act like the most similar good state
        idx = max(range(len(good_states)), key=lambda i: cosine_sim(state, good_states[i]))
        return good_actions[idx]
    else:
        # Act opposite of the most similar bad state
        idx = max(range(len(bad_states)), key=lambda i: cosine_sim(state, bad_states[i]))
        return not bad_actions[idx]

# ── Tkinter ───────────────────────────────────────────────────
root = tk.Tk()
root.title("TSO Snail Jumper")
canvas = tk.Canvas(root, width=SCREEN_W, height=SCREEN_H, bg='#1a1a2e')
canvas.pack()

# Intro
canvas.create_text(SCREEN_W//2, 80, text="🧠 TSO Snail Jumper",
                   fill='#4fc3f7', font=('Arial', 36, 'bold'))
canvas.create_text(SCREEN_W//2, 120, text="Topographic Stabilization Operator",
                   fill='#666', font=('Arial', 13))
canvas.create_text(SCREEN_W//2, 180,
                   text=f"TSO a vu {stats['n_games']} parties en {stats['training_time']}s",
                   fill='#e0e0e0', font=('Arial', 16))
canvas.create_text(SCREEN_W//2, 210,
                   text=f"Memoire: {stats['n_good']} etats bons + {stats['n_bad']} etats mortels",
                   fill='#aaa', font=('Arial', 13))
canvas.create_text(SCREEN_W//2, 240,
                   text=f"Decision: k-NN cosinus  |  Precision: {stats.get('accuracy', '?')}%",
                   fill='#aaa', font=('Arial', 13))
canvas.create_text(SCREEN_W//2, 280,
                   text="0 backprop  •  0 GPU  •  Memoire geometrique",
                   fill='#555', font=('Arial', 11))

btn = tk.Button(root, text="▶  GO — Lancer le jeu",
                font=('Arial', 18, 'bold'),
                bg='#4fc3f7', fg='#1a1a2e',
                command=lambda: start_game())
btn.place(x=SCREEN_W//2 - 130, y=330, width=260, height=50)
root.bind('<Escape>', lambda e: root.destroy())

def start_game():
    btn.destroy()
    px = SCREEN_W / 2
    obstacles = []
    score = 0
    frame = 0
    alive = True
    sim_val = 0.0
    paused = False

    def on_key(e):
        nonlocal paused
        if e.keysym == 'r':
            px = SCREEN_W / 2; obstacles.clear(); score = 0; frame = 0; alive = True
        if e.keysym == 'p': paused = not paused
        if e.keysym == 'Escape': root.destroy()
    root.bind('<Key>', on_key)

    def update():
        nonlocal px, obstacles, score, frame, alive, sim_val, paused

        if not paused and alive:
            frame += 1
            state = encode(px, obstacles)
            left = decide(state)
            px = max(min(px + (-5.0 if left else 5.0), SCREEN_W - 10.0), 10.0)

            speed = 3.0 + frame * 0.002
            si = max(int(40 - speed * 2), 8)
            if frame % si == 0:
                ox = px + random.uniform(-80, 80)
                ox = max(10, min(SCREEN_W - 10, ox))
                obstacles.append({'x': ox, 'y': -20})
            for o in obstacles:
                o['y'] += speed
            obstacles[:] = [o for o in obstacles if o['y'] < SCREEN_H + 20]

            # Compute sim for display
            if good_states:
                sim_val = max(cosine_sim(state, s) for s in good_states)

            for o in obstacles:
                if abs(px - o['x']) < 21 and abs(PLAYER_Y - o['y']) < 22:
                    alive = False
                    break
            score = frame // 6

        canvas.delete('all')
        canvas.create_rectangle(0, PLAYER_Y + 30, SCREEN_W, SCREEN_H, fill='#2d2d3f', outline='')
        c = '#4fc3f7' if alive else '#ef5350'
        px_i = int(px)
        canvas.create_rectangle(px_i - 15, PLAYER_Y - 15, px_i + 15, PLAYER_Y + 15, fill=c, outline='#29b6f6' if alive else '#e53935', width=2)
        for o in obstacles:
            ox, oy = int(o['x']), int(o['y'])
            canvas.create_rectangle(ox - 11, oy, ox + 11, oy + 20, fill='#ef5350', outline='#c62828')

        canvas.create_text(SCREEN_W//2, 20, text=f"Score: {score}", fill='#e0e0e0', font=('Arial', 18, 'bold'))
        sc = '#4fc3f7' if alive else '#ef5350'
        canvas.create_text(SCREEN_W - 80, 20, text='ALIVE' if alive else 'DEAD', fill=sc, font=('Arial', 12, 'bold'))
        canvas.create_text(10, 50, text=f"k-NN sim: {sim_val:.4f}  protos: {len(good_states)}+{len(bad_states)}",
                          fill='#aaa', font=('Arial', 10), anchor='w')
        canvas.create_text(10, 65, text=f"Frame: {frame}  Score: {score}",
                          fill='#666', font=('Arial', 9), anchor='w')
        canvas.create_text(SCREEN_W//2, SCREEN_H - 15, text="[P]ause [R]eset [Esc]uit", fill='#555', font=('Arial', 9))

        root.after(16, update)

    root.after(16, update)

root.mainloop()
