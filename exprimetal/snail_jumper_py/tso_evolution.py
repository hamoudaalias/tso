"""
TSO Neuroevolution for Snail Jumper.
Population of TSOAgent — crossover + mutation on prototype vectors.
"""
import copy
import numpy as np
from tso_agent import TSOAgent, AttractorField

POP_SIZE = 100
EMBED_DIM = 8
GENERATIONS = 50
MAX_FRAMES = 5000

def simulate_agent(agent):
    """Run one episode, return fitness (score)."""
    game = {
        'player_x': 250.0,
        'screen_w': 500.0,
        'screen_h': 700.0,
        'obstacles': [],
        'alive': True,
        'frame': 0,
        'score': 0,
        'spawn_timer': 0,
    }
    import random

    while game['alive'] and game['frame'] < MAX_FRAMES:
        game['frame'] += 1

        left = agent.decide(game)
        if left:
            game['player_x'] = max(game['player_x'] - 4.0, 10.0)
        else:
            game['player_x'] = min(game['player_x'] + 4.0, game['screen_w'] - 10.0)

        game['spawn_timer'] += 1
        if game['spawn_timer'] > 20 + random.randint(0, 20):
            game['spawn_timer'] = 0
            game['obstacles'].append({
                'x': random.uniform(10.0, game['screen_w'] - 10.0),
                'y': -20.0,
            })

        speed = 3.0
        for obs in game['obstacles']:
            obs['y'] += speed
        game['obstacles'] = [o for o in game['obstacles'] if o['y'] < game['screen_h'] + 20.0]

        for obs in game['obstacles']:
            dx = game['player_x'] - obs['x']
            dy = (game['screen_h'] - 50.0) - obs['y']
            if abs(dx) < 15.0 and abs(dy) < 20.0:
                game['alive'] = False
                agent.learn(game, False)
                break
        else:
            agent.learn(game, True)

        game['score'] = game['frame'] // 6

    return game['score']

def crossover(p1, p2):
    field = AttractorField(EMBED_DIM, 0, 0, 0.05)
    for c in range(min(len(p1.field.prototypes), len(p2.field.prototypes))):
        protos1 = p1.field.prototypes[c]
        protos2 = p2.field.prototypes[c]
        half = len(protos1) // 2
        child_protos = protos1[:half].copy()
        child_protos += protos2[half:half + len(protos2) - half].copy() if half < len(protos2) else []
        if not child_protos:
            continue
        field.add_class(child_protos[0])
        for p in child_protos[1:]:
            field.add_prototype(p, c)
    agent = TSOAgent(EMBED_DIM)
    agent.field = field
    return agent

def mutate(agent):
    for protos in agent.field.prototypes:
        for i in range(len(protos)):
            if np.random.random() < 0.3:
                noise = np.random.uniform(-0.1, 0.1, agent.embed_dim)
                protos[i] = protos[i] + noise
                n = max(np.linalg.norm(protos[i]), 1e-12)
                protos[i] /= n
    return agent

class Evolution:
    def __init__(self):
        self.agents = [TSOAgent(EMBED_DIM) for _ in range(POP_SIZE)]
        self.generation = 0
        self.best_fitness = 0

    def evaluate(self):
        for agent in self.agents:
            agent.fitness = simulate_agent(agent)
            if agent.fitness > self.best_fitness:
                self.best_fitness = agent.fitness

    def next_generation(self):
        self.evaluate()

        avg = np.mean([a.fitness for a in self.agents])
        print(f"Gen {self.generation:>3} | best={self.agents[0].fitness:>4} | avg={avg:.1f}")

        survivors = self.agents[:POP_SIZE // 4]
        next_gen = []
        while len(next_gen) < POP_SIZE:
            p1 = survivors[np.random.randint(len(survivors))]
            p2 = survivors[np.random.randint(len(survivors))]
            child = crossover(p1, p2)
            child = mutate(child)
            next_gen.append(child)

        self.agents = next_gen
        self.generation += 1

    def run(self):
        for _ in range(GENERATIONS):
            self.next_generation()

if __name__ == '__main__':
    evo = Evolution()
    evo.run()
    print(f"\nBest fitness: {evo.best_fitness}")
