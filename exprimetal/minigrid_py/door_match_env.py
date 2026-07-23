"""
MiniGrid DoorMatch-v0 — Test mémoire épisodique.
Portes de choix sur des RANGÉES DIFFÉRENTES pour navigation garantie.
"""
import random
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Wall
from minigrid.minigrid_env import MiniGridEnv


class DoorMatchEnv(MiniGridEnv):
    def __init__(self, max_steps=80, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=9,
            height=13,
            max_steps=max_steps,
            **kwargs,
        )
        self.start_color = None
        self.start_door = None
        self.left_door = None
        self.right_door = None

    @staticmethod
    def _gen_mission():
        return "open the matching colour door"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.start_color = random.choice(["red", "blue"])
        other = "blue" if self.start_color == "red" else "red"

        # Layout 9×13:
        #  0: # # # # # # # # #
        #  1: # . . . . . . . #
        #  2: # D . . . . . . #
        #  3: # . . . . . . . #
        #  4: # . . . . . . . #
        #  5: # . . . . . . . #
        #  6: # # # # # . . . #
        #  7: # . . . . . . . #
        #  8: # . . . . . . . #
        #  9: # # # # # # # . #
        # 10: # . . . . . . . #
        # 11: # R . . . . . B #
        # 12: # # # # # # # # #
        #
        # R et B randomisés. L'une correspond à start_color.
        # Chemin: agent traverse corridor, arrive en (7,10)↓,
        # voit porte B en (7,11): si match→toggle, sinon→va à R en (1,11).

        self.start_door = Door(self.start_color, is_locked=False)
        self.grid.set(1, 3, self.start_door)
        self.agent_pos = (1, 4)
        self.agent_dir = 3  # up

        # Corridor droit bloqué par occlusion partielle
        for x in range(1, 6):
            self.grid.set(x, 6, Wall())
        for x in range(1, 7):
            self.grid.set(x, 9, Wall())

        # Portes de choix
        a = Door(self.start_color, is_locked=False)
        b = Door(other, is_locked=False)
        if random.random() < 0.5:
            self.left_door, self.right_door = a, b
        else:
            self.left_door, self.right_door = b, a
        self.grid.set(1, 11, self.left_door)
        self.grid.set(7, 11, self.right_door)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            ax, ay = self.agent_pos
            ad = self.agent_dir
            dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            fx = ax + dirs[ad][0]
            fy = ay + dirs[ad][1]
            cell = self.grid.get(fx, fy)
            if cell is self.left_door or cell is self.right_door:
                if cell.color == self.start_color:
                    reward = 1.0
                else:
                    reward = -0.5
                terminated = True

        return obs, reward, terminated, truncated, info
