"""
TSO Snail Jumper Visualizer.
Loads a trained TSO agent (from Rust) and plays the game with pygame.
"""
import json
import sys
import math
import random

try:
    import pygame
except ImportError:
    print("Install pygame: pip install pygame")
    sys.exit(1)

SCREEN_W = 604
SCREEN_H = 800

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
    best_class = 0
    best_sim = -1
    for c, protos in enumerate(prototypes):
        for p in protos:
            s = cosine_sim(state, p)
            if s > best_sim:
                best_sim = s
                best_class = c
    return best_class, best_sim

def encode(player_x, obstacles, screen_w=SCREEN_W, screen_h=SCREEN_H):
    v = [0.0] * 8
    v[0] = player_x / screen_w
    if obstacles:
        nearest = min(obstacles, key=lambda o: (screen_h - o['y']) if o['y'] > -50 else float('inf'))
        v[1] = nearest['x'] / screen_w
        v[2] = nearest['y'] / screen_h if nearest['y'] > 0 else 0.0
        v[3] = (nearest['x'] - player_x) / screen_w
    v[4] = len(obstacles) / 10.0
    for i, o in enumerate(obstacles[:3]):
        v[5 + i] = min(max(o['y'] / screen_h, 0.0), 1.0)
    return v

def draw_text(surf, text, size, x, y, color=(255, 255, 255)):
    font = pygame.font.SysFont('Arial', size)
    img = font.render(str(text), True, color)
    surf.blit(img, (x, y))

def main():
    if len(sys.argv) > 1:
        agent_path = sys.argv[1]
    else:
        agent_path = "best_agent.json"

    try:
        prototypes = load_agent(agent_path)
    except FileNotFoundError:
        print(f"Agent file '{agent_path}' not found.")
        print("Run 'cargo run -p tso-snail-jumper --release' first to train.")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("TSO Snail Jumper — Visual Play")
    clock = pygame.time.Clock()

    player_x = SCREEN_W / 2
    obstacles = []
    score = 0
    speed = 4.0
    frame = 0
    lif_state = [0.0] * 8
    alpha = 0.8
    alive = True
    paused = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    player_x = SCREEN_W / 2
                    obstacles = []
                    score = 0
                    frame = 0
                    lif_state = [0.0] * 8
                    alive = True

        if not paused and alive:
            frame += 1

            state = encode(player_x, obstacles)
            lif_state = [alpha * s + (1 - alpha) * x for s, x in zip(lif_state, state)]

            cls, sim = predict(lif_state, prototypes)
            left = cls == 0

            if left:
                player_x = max(player_x - 4.0, 10.0)
            else:
                player_x = min(player_x + 4.0, SCREEN_W - 10.0)

            if random.randint(0, 25) == 0:
                obstacles.append({'x': random.uniform(30, SCREEN_W - 30), 'y': -20})

            for o in obstacles:
                o['y'] += speed
            obstacles = [o for o in obstacles if o['y'] < SCREEN_H + 20]

            for o in obstacles:
                dx = player_x - o['x']
                dy = (SCREEN_H - 80) - o['y']
                if abs(dx) < 15 and abs(dy) < 20:
                    alive = False

            score = frame // 6

        screen.fill((240, 240, 240))

        # Player (snail)
        snail_color = (50, 180, 50) if alive else (180, 50, 50)
        pygame.draw.rect(screen, snail_color, (player_x - 15, SCREEN_H - 100, 30, 30))
        pygame.draw.circle(screen, (30, 120, 30) if alive else (120, 30, 30),
                           (int(player_x), SCREEN_H - 100), 18, 3)

        # Obstacles
        for o in obstacles:
            pygame.draw.rect(screen, (200, 50, 50), (o['x'] - 10, o['y'], 20, 20))

        # Ground
        pygame.draw.line(screen, (100, 100, 100), (0, SCREEN_H - 70), (SCREEN_W, SCREEN_H - 70), 2)

        # HUD
        draw_text(screen, f"Score: {score}", 20, 10, 10, (30, 30, 30))
        draw_text(screen, f"Alive: {'YES' if alive else 'NO'}", 16, 10, 35,
                  (30, 150, 30) if alive else (150, 30, 30))
        draw_text(screen, f"LIF: [{lif_state[0]:.2f} {lif_state[1]:.2f} {lif_state[2]:.2f} {lif_state[3]:.2f}]", 14, 10, 55, (80, 80, 80))
        draw_text(screen, f"LVQ1 sim: {sim:.4f}  action: {'LEFT' if left else 'RIGHT'}", 14, 10, 75, (80, 80, 80))
        draw_text(screen, "[SPACE] pause  [R] reset  [ESC] quit", 12, 10, SCREEN_H - 20, (120, 120, 120))

        # LIF bar
        bar_x, bar_y = 10, 100
        bar_w = 200
        bar_h = 12
        for i in range(min(8, len(lif_state))):
            val = max(-1, min(1, lif_state[i]))
            color = (50 + int(200 * (val + 1) / 2), 100, 200 - int(200 * (val + 1) / 2))
            pygame.draw.rect(screen, (200, 200, 200), (bar_x + i * (bar_w // 8), bar_y, bar_w // 8 - 1, bar_h))
            fill_w = int((bar_w // 8 - 1) * abs(val))
            fill_x = bar_x + i * (bar_w // 8)
            if val > 0:
                pygame.draw.rect(screen, color, (fill_x + (bar_w // 8 - 1) // 2, bar_y, fill_w, bar_h))
            else:
                pygame.draw.rect(screen, color, (fill_x + (bar_w // 8 - 1) // 2 - fill_w, bar_y, fill_w, bar_h))

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
