"""
Encodeur symbolique MiniGrid.
Extrait les objets visibles et les encode en vecteur structuré,
comme des "mots" dans le pipeline NLP de TSO.
"""
import numpy as np

MAX_OBJ = 5
INTERESTING_TYPES = {4, 5, 6, 7, 8}
DIR_MAP = {0: "right", 1: "down", 2: "left", 3: "up"}
OBJ_NAMES = {4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal"}
COL_NAMES = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}


class SymbolicEncoder:
    """Encode l'observation MiniGrid en vecteur d'états symboliques."""

    def __init__(self):
        self.dim = MAX_OBJ * 5 + 4

    def encode(self, obs):
        img = obs["image"]
        direction = obs["direction"]

        obj_vecs = []
        for y in range(7):
            for x in range(7):
                t, c, s = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                if t not in INTERESTING_TYPES:
                    continue
                dx, dy = (x - 3) / 7.0, (y - 3) / 7.0
                obj_vecs.append([
                    t / 10.0,
                    c / 6.0,
                    s / 3.0,
                    dx,
                    dy,
                ])

        obj_vecs.sort(key=lambda v: abs(v[3]) + abs(v[4]))
        obj_vecs = obj_vecs[:MAX_OBJ]

        state = np.zeros(self.dim, dtype=np.float64)
        for i, vec in enumerate(obj_vecs):
            state[i * 5 : (i + 1) * 5] = vec

        dir_offset = MAX_OBJ * 5
        state[dir_offset + direction] = 1.0

        return state

    def describe(self, obs):
        """Version lisible pour debug."""
        img = obs["image"]
        items = []
        for y in range(7):
            for x in range(7):
                t, c, s = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
                if t not in INTERESTING_TYPES:
                    continue
                dx, dy = x - 3, y - 3
                name = OBJ_NAMES.get(t, f"t{t}")
                col = COL_NAMES.get(c, f"c{c}")
                items.append(f"({dx:+d},{dy:+d}) {col} {name} state={s}")
        return items
