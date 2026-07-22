"""
Snail Jumper with TSO Brain — fully self-contained pygame game.
No neural network, no backprop — TSO cognitive engine (LIF + LVQ1).
"""
import sys
import os
import random
import math

import pygame

from tso_agent import TSOAgent, AttractorField

SCREEN_W = 604
SCREEN_H = 800
POP_SIZE = 100

# ── TSO Evolution ─────────────────────────────────────────────

def simulate_one(agent, max_frames=5000):
    game_state = {
        'player_x': SCREEN_W / 2,
        'screen_w': SCREEN_W,
        'screen_h': SCREEN_H,
        'obstacles': [],
        'alive': True,
        'frame': 0,
        'spawn_timer': 0,
    }
    speed = 3.0
    while game_state['alive'] and game_state['frame'] < max_frames:
        game_state['frame'] += 1
        left = agent.decide(game_state)
        game_state['player_x'] = max(min(
            game_state['player_x'] + (-4.0 if left else 4.0),
            SCREEN_W - 10.0), 10.0)
        game_state['spawn_timer'] += 1
        if game_state['spawn_timer'] > 20 + random.randint(0, 20):
            game_state['spawn_timer'] = 0
            game_state['obstacles'].append({
                'x': random.uniform(10.0, SCREEN_W - 10.0),
                'y': -20.0})
        for o in game_state['obstacles']:
            o['y'] += speed
        game_state['obstacles'] = [o for o in game_state['obstacles'] if o['y'] < SCREEN_H + 20]
        hit = False
        for o in game_state['obstacles']:
            dx = game_state['player_x'] - o['x']
            dy = (SCREEN_H - 50) - o['y']
            if abs(dx) < 15 and abs(dy) < 20:
                game_state['alive'] = False
                hit = True
                break
        agent.learn(game_state, not hit)
    return game_state['frame'] // 6


def crossover(p1, p2):
    field = AttractorField(8, 0, 0, 0.05)
    for c in range(min(len(p1.field.prototypes), len(p2.field.prototypes))):
        protos1 = p1.field.prototypes[c]
        protos2 = p2.field.prototypes[c]
        half = len(protos1) // 2
        child_protos = protos1[:half].copy()
        if half < len(protos2):
            child_protos += protos2[half: half + len(protos2) - half].copy()
        if not child_protos:
            continue
        if field.n_classes() <= c:
            field.add_class(child_protos[0])
        for p in child_protos[1:]:
            field.add_prototype(p, c)
    agent = TSOAgent(8)
    agent.field = field
    return agent


def mutate(agent):
    for protos in agent.field.prototypes:
        for i in range(len(protos)):
            if random.random() < 0.3:
                noise = [random.uniform(-0.1, 0.1) for _ in range(8)]
                protos[i] = [a + b for a, b in zip(protos[i], noise)]
                n = math.sqrt(sum(x * x for x in protos[i])) + 1e-12
                protos[i] = [x / n for x in protos[i]]
    return agent


class TSOEvolution:
    def __init__(self):
        self.agents = [TSOAgent(8) for _ in range(POP_SIZE)]
        self.generation = 0
        self.best_fitness = 0
        self.best_agent = self.agents[0]
        self.all_scores = []

    def step(self):
        for agent in self.agents:
            agent.fitness = simulate_one(agent)
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        if self.agents[0].fitness > self.best_fitness:
            self.best_fitness = self.agents[0].fitness
            self.best_agent = self.agents[0]
        self.all_scores.append(self.agents[0].fitness)
        survivors = self.agents[:POP_SIZE // 4]
        next_gen = []
        while len(next_gen) < POP_SIZE:
            p1 = survivors[random.randint(0, len(survivors) - 1)]
            p2 = survivors[random.randint(0, len(survivors) - 1)]
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)
        self.agents = next_gen
        self.generation += 1
        avg = sum(a.fitness for a in survivors) / len(survivors)
        return self.agents[0].fitness, avg


# ── Pygame Game ───────────────────────────────────────────────

def draw_text(surf, text, size, x, y, color=(255, 255, 255)):
    font = pygame.font.SysFont('Arial', size)
    img = font.render(text, True, color)
    surf.blit(img, (x, y))


def run_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("TSO Snail Jumper")
    clock = pygame.time.Clock()

    evo = TSOEvolution()
    playing = False
    generation = 0
    player_x = SCREEN_W / 2
    obstacles = []
    score = 0
    tso_agent = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if event.key == pygame.K_SPACE and playing:
                    pass  # manual mode not needed
            if event.type == pygame.MOUSEBUTTONDOWN and not playing:
                mx, my = pygame.mouse.get_pos()
                if 150 <= mx <= 450 and 350 <= my <= 400:
                    playing = True
                    generation = 0
                    evo = TSOEvolution()
                    tso_agent = evo.best_agent
                    player_x = SCREEN_W / 2
                    obstacles = []
                    score = 0

        if not playing:
            screen.fill((47, 72, 88))
            draw_text(screen, "TSO Snail Jumper", 50, SCREEN_W // 2 - 180, 80, (111, 196, 169))
            draw_text(screen, "No neural network. No backprop.", 20, SCREEN_W // 2 - 140, 140, (200, 200, 200))
            draw_text(screen, "TSO engine: LIF + LVQ1 prototypes", 20, SCREEN_W // 2 - 140, 170, (200, 200, 200))
            pygame.draw.rect(screen, (111, 196, 169), (150, 350, 300, 50), 2)
            draw_text(screen, "Start TSO Neuroevolution", 25, 175, 360, (111, 196, 169))
            pygame.display.flip()
            clock.tick(60)
            continue

        if generation < 50:
            best, avg = evo.step()
            generation += 1
            tso_agent = evo.best_agent
            print(f"Gen {generation:>3} | best={best:>4} | avg={avg:.1f}")

        if tso_agent is None:
            continue

        # Play with best agent
        speed = 3 + generation * 0.05
        game_state = {
            'player_x': player_x, 'screen_w': SCREEN_W, 'screen_h': SCREEN_H,
            'obstacles': [{'x': o['x'], 'y': o['y']} for o in obstacles],
            'alive': True, 'frame': 0, 'spawn_timer': 0,
        }

        left = tso_agent.decide(game_state)
        player_x = max(min(player_x + (-4.0 if left else 4.0), SCREEN_W - 10.0), 10.0)

        if random.randint(0, 25) == 0:
            obstacles.append({'x': random.uniform(30, SCREEN_W - 30), 'y': -20})

        for o in obstacles:
            o['y'] += speed
        obstacles = [o for o in obstacles if o['y'] < SCREEN_H + 20]

        for o in obstacles:
            dx = player_x - o['x']
            dy = (SCREEN_H - 80) - o['y']
            if abs(dx) < 15 and abs(dy) < 20:
                pass  # just show, don't die in demo mode

        score += 1

        # Draw
        screen.fill((255, 255, 255))
        # Player
        pygame.draw.rect(screen, (50, 150, 50), (player_x - 15, SCREEN_H - 100, 30, 30))
        pygame.draw.circle(screen, (30, 100, 30), (int(player_x), SCREEN_H - 100), 18, 3)
        # Obstacles
        for o in obstacles:
            pygame.draw.rect(screen, (200, 50, 50), (o['x'] - 10, o['y'], 20, 20))
            pygame.draw.circle(screen, (150, 30, 30), (int(o['x']), int(o['y'])), 12, 2)

        # TSO brain info
        state = np_state = tso_agent.encode(game_state)
        cls, dist = tso_agent.field.predict(tso_agent.lif.state.tolist())

        draw_text(screen, f"Gen: {generation}/50", 18, 10, 10, (30, 30, 30))
        draw_text(screen, f"Best: {evo.best_fitness}", 18, 10, 35, (30, 30, 30))
        draw_text(screen, f"Score: {score}", 18, 10, 60, (30, 30, 30))
        draw_text(screen, f"LVQ1 class: {cls} (dist={dist:.3f})", 14, 10, 85, (100, 100, 100))
        draw_text(screen, f"LIF dim 0-3: {tso_agent.lif.state[0]:.2f} {tso_agent.lif.state[1]:.2f} {tso_agent.lif.state[2]:.2f} {tso_agent.lif.state[3]:.2f}", 12, 10, 105, (120, 120, 120))
        draw_text(screen, f"Action: {'LEFT' if left else 'RIGHT'}", 16, 10, 125, (80, 80, 200))
        draw_text(screen, f"ESC: quit", 12, SCREEN_W - 100, SCREEN_H - 30, (100, 100, 100))

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    run_game()
