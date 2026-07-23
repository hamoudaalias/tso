import numpy as np
from tso_pyo3 import AttractorField


def extract_blocks(rgb, grid=8):
    h, w = rgb.shape[:2]
    gray = np.mean(rgb, axis=2).astype(np.float64)
    ex = np.abs(np.diff(gray, axis=1))
    ey = np.abs(np.diff(gray, axis=0))
    edges = np.zeros_like(gray)
    edges[:, :-1] += ex
    edges[:-1, :] += ey
    bs = h // grid
    blocks = []
    for i in range(grid):
        for j in range(grid):
            y, x = i * bs, j * bs
            b = rgb[y:y + bs, x:x + bs].astype(np.float64) / 255.0
            eb = edges[y:y + bs, x:x + bs] / 255.0
            blocks.append([eb.mean()] + b.mean(axis=(0, 1)).tolist())
    return np.array(blocks, dtype=np.float64)


def develop_v1(env, n_steps=1000, n_protos=32, lr=0.05):
    field = None
    init = False
    dim = 4
    for step in range(n_steps):
        obs = env.reset()
        blks = extract_blocks(obs)
        for b in blks:
            if not init:
                if field is None:
                    field = AttractorField(dim, n_protos, 3, lr)
                field.add_class(b.tolist())
                init = field.n_classes() >= n_protos
            else:
                w = field.predict(b.tolist())
                field.train_step(b.tolist(), w)
    return field


def encode_v1(field, rgb):
    blks = extract_blocks(rgb)
    vec = np.zeros(64, dtype=np.float64)
    n = field.n_classes()
    for i, b in enumerate(blks):
        vec[i] = field.predict(b.tolist()) / max(n - 1, 1)
    return vec.tolist()
