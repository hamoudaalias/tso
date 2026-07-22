"""
TSO Snail Jumper — Tkinter visualizer (no pygame needed).
Shows the TSO brain (LIF + LVQ1) playing in real time.
"""
import json
import sys
import math
import random
import time
import tkinter as tk

SCREEN_W = 600
SCREEN_H = 700
PLAYER_Y = SCREEN_H - 100

def load_agent(path="best_agent.json"):
    with open(path) as f:
        data = json.load(f)
    return data['prototypes']

def cosine_sim(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-12
    nb = math.sqrt(sum(x*x for x in b)) + 1e-12
    return dot / (na * nb)

def predict(state, prototypes):
    best_cls = 0
    best_sim = -1
    for c, protos in enumerate(prototypes):
        for p in protos:
            s = cosine_sim(state, p)
            if s > best_sim:
                best_sim = s
                best_cls = c
    return best_cls, best_sim

def encode(px, obstacles):
    v = [0.0] * 8
    v[0] = px / SCREEN_W
    if obstacles:
        nearest = min(obstacles, key=lambda o: o['y'] if o['y'] > -50 else float('inf'))
        v[1] = nearest['x'] / SCREEN_W
        v[2] = max(nearest['y'] / SCREEN_H, 0.0)
        v[3] = (nearest['x'] - px) / SCREEN_W
    v[4] = len(obstacles) / 10.0
    for i, o in enumerate(obstacles[:3]):
        v[5 + i] = min(max(o['y'] / SCREEN_H, 0.0), 1.0)
    return v


def main():
    agent_path = sys.argv[1] if len(sys.argv) > 1 else "best_agent.json"
    try:
        prototypes = load_agent(agent_path)
    except FileNotFoundError:
        print(f"Agent file '{agent_path}' not found. Train first: cargo run -p tso-snail-jumper --release")
        sys.exit(1)

    root = tk.Tk()
    root.title("TSO Snail Jumper")
    canvas = tk.Canvas(root, width=SCREEN_W, height=SCREEN_H, bg='#f0f0f0')
    canvas.pack()

    px = SCREEN_W / 2
    obstacles = []
    score = 0
    frame = 0
    alive = True
    alpha = 0.8
    lif = [0.0] * 8
    last_action = "RIGHT"
    sim_val = 0.0
    paused = False

    def reset_game():
        nonlocal px, obstacles, score, frame, alive, lif
        px = SCREEN_W / 2
        obstacles.clear()
        score = 0
        frame = 0
        alive = True
        lif = [0.0] * 8

    def on_key(e):
        nonlocal paused
        if e.keysym == 'r':
            reset_game()
        if e.keysym == 'p':
            paused = not paused
        if e.keysym == 'Escape':
            root.destroy()

    root.bind('<Key>', on_key)

    def update():
        nonlocal px, obstacles, score, frame, alive, lif, last_action, sim_val

        if not paused and alive:
            frame += 1

            state = encode(px, obstacles)
            lif = [alpha * s + (1 - alpha) * x for s, x in zip(lif, state)]

            cls, sim_val = predict(lif, prototypes)
            last_action = "LEFT" if cls == 0 else "RIGHT"

            if cls == 0:
                px = max(px - 4.0, 10.0)
            else:
                px = min(px + 4.0, SCREEN_W - 10.0)

            if random.randint(0, 25) == 0:
                obstacles.append({'x': random.uniform(30, SCREEN_W - 30), 'y': -20})

            for o in obstacles:
                o['y'] += 4.0
            obstacles[:] = [o for o in obstacles if o['y'] < SCREEN_H + 20]

            for o in obstacles:
                dx = px - o['x']
                dy = PLAYER_Y - o['y']
                if abs(dx) < 15 and abs(dy) < 20:
                    alive = False

            score = frame // 6

        canvas.delete('all')

        # Ground
        canvas.create_line(0, PLAYER_Y + 30, SCREEN_W, PLAYER_Y + 30, fill='#888', width=2)

        # Player
        px_int = int(px)
        py = PLAYER_Y
        color = '#2e8b2e' if alive else '#cc3333'
        canvas.create_rectangle(px_int - 15, py - 15, px_int + 15, py + 15, fill=color, outline='#1a5c1a' if alive else '#992222')
        canvas.create_oval(px_int - 18, py - 18, px_int + 18, py + 18, outline='#1a5c1a' if alive else '#992222', width=2)

        # Obstacles
        for o in obstacles:
            ox = int(o['x'])
            oy = int(o['y'])
            canvas.create_rectangle(ox - 10, oy, ox + 10, oy + 20, fill='#cc3333', outline='#992222')
            canvas.create_text(ox, oy + 10, text='X', fill='white', font=('Arial', 10, 'bold'))

        # HUD
        canvas.create_text(100, 20, text=f"Score: {score}", fill='#222', font=('Arial', 16, 'bold'))
        canvas.create_text(100, 45, text=f"{'ALIVE' if alive else 'DEAD'}", fill='#2e8b2e' if alive else '#cc3333', font=('Arial', 14, 'bold'))
        canvas.create_text(100, 70, text=f"LIF: [{lif[0]:.2f} {lif[1]:.2f} {lif[2]:.2f} {lif[3]:.2f}]", fill='#555', font=('Arial', 10))
        canvas.create_text(100, 90, text=f"LVQ1 sim: {sim_val:.4f}  Action: {last_action}", fill='#555', font=('Arial', 10))
        canvas.create_text(100, 110, text=f"Frames: {frame}  Protos: 2x{len(prototypes[0])}", fill='#555', font=('Arial', 10))
        canvas.create_text(SCREEN_W - 120, SCREEN_H - 15, text="[P]ause [R]eset [Esc]uit", fill='#999', font=('Arial', 9))

        # LIF bar
        bar_x, bar_y = 10, 130
        bw = 180
        bh = 10
        for i in range(8):
            val = max(-1, min(1, lif[i]))
            r = 50 + int(200 * (val + 1) / 2)
            g = 100
            b = 200 - int(200 * (val + 1) / 2)
            fill = f'#{r:02x}{g:02x}{b:02x}'
            seg_x = bar_x + i * (bw // 8)
            canvas.create_rectangle(seg_x, bar_y, seg_x + bw // 8 - 1, bar_y + bh, fill='#ddd', outline='')
            if val > 0:
                fw = int((bw // 8 - 1) * abs(val))
                canvas.create_rectangle(seg_x + (bw // 8 - 1) // 2, bar_y, seg_x + (bw // 8 - 1) // 2 + fw, bar_y + bh, fill=fill, outline='')
            else:
                fw = int((bw // 8 - 1) * abs(val))
                canvas.create_rectangle(seg_x + (bw // 8 - 1) // 2 - fw, bar_y, seg_x + (bw // 8 - 1) // 2, bar_y + bh, fill=fill, outline='')

        root.after(16, update)

    root.after(16, update)
    root.mainloop()

if __name__ == '__main__':
    main()
