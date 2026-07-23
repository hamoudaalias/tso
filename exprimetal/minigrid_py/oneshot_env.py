"""
MiniGrid OneShot-v0 — Test pur de mémoire de travail pour TSO.
Avec randomisation des POSITIONS (matching à gauche ou à droite).
"""
import random
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

COLORS = ["red", "green", "blue", "yellow"]
TYPES = [("ball", Ball), ("key", Key)]


class OneShotEnv(MiniGridEnv):
    def __init__(self, width=15, height=6, max_steps=60, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            **kwargs,
        )
        self.target_type = None
        self.target_color = None
        self.match_pos = None
        self.dist_pos = None

    @staticmethod
    def _gen_mission():
        return "pick up the matching object"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        t_type, t_cls = random.choice(TYPES)
        d_type, d_cls = random.choice([x for x in TYPES if x[0] != t_type])
        t_color = random.choice(COLORS)
        d_color = random.choice([c for c in COLORS if c != t_color])
        self.target_type = t_type
        self.target_color = t_color

        # Murs épais (row 1 accessible depuis la gauche pour le contournement)
        for x in [4, 5]:
            for y in [1, 2]:
                self.grid.set(x, y, Wall())
        for x in [10, 11]:
            for y in [2]:
                self.grid.set(x, y, Wall())

        self.grid.set(12, 2, Wall())

        # Agent
        self.agent_pos = (2, 4)
        self.agent_dir = 0

        # Cible
        self.put_obj(t_cls(t_color), 3, 1)

        # Matching et leurre: POSITIONS RANDOMISÉES (gauche/droite aléatoire)
        if random.random() < 0.5:
            self.put_obj(t_cls(t_color), 13, 1)
            self.put_obj(d_cls(d_color), 13, 3)
            self.match_pos = (13, 1)
            self.dist_pos = (13, 3)
        else:
            self.put_obj(t_cls(t_color), 13, 3)
            self.put_obj(d_cls(d_color), 13, 1)
            self.match_pos = (13, 3)
            self.dist_pos = (13, 1)

        self.mission = "pick up the matching object"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if action == self.actions.pickup:
            if self.carrying:
                if (self.carrying.type == self.target_type and 
                    self.carrying.color == self.target_color):
                    reward = 1.0
                    terminated = True
                else:
                    reward = -1.0
                    terminated = True
        return obs, reward, terminated, truncated, info
