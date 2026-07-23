"""
MiniGrid KeyMatch-v0 — Test WorkingMemory : clé → porte verrouillée → but.
"""
import random
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class KeyMatchEnv(MiniGridEnv):
    def __init__(self, max_steps=50, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=9,
            height=5,
            max_steps=max_steps,
            **kwargs,
        )
        self.key_color = None
        self.key_obj = None
        self.locked_door = None
        self.ball = None

    @staticmethod
    def _gen_mission():
        return "pick up the ball"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Palette de couleurs pour la clé
        color = random.choice(["red", "blue", "green", "yellow", "purple", "grey"])

        # Layout 9×5:
        # 0: # # # # # # # # #
        # 1: # K . . . . L . #
        # 2: # . # # # # # . #
        # 3: # . . . . . . B #
        # 4: # # # # # # # # #

        # Clé à (1,1)
        self.key_color = color
        self.key_obj = Key(color)
        self.grid.set(1, 1, self.key_obj)

        # Porte verrouillée à (6,1) — même couleur que la clé
        self.locked_door = Door(color, is_locked=True)
        self.grid.set(6, 1, self.locked_door)

        # Ball-but à (7,3)
        self.ball = Ball("green")
        self.grid.set(7, 3, self.ball)

        # Occlusion partielle en y=2, x=2..6
        for x in range(2, 7):
            self.grid.set(x, 2, Wall())

        # Agent en (1,2) face à la clé (up)
        self.agent_pos = (1, 2)
        self.agent_dir = 3  # up

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Succès si l'agent porte la balle
        if action == self.actions.pickup:
            if self.carrying is self.ball:
                reward = 1.0
                terminated = True

        return obs, reward, terminated, truncated, info
