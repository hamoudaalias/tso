"""
Self-Organizing Map for topographical clustering of word embeddings.
"""
import numpy as np
import math


class SOM:
    """Self-Organizing Map with Gaussian neighbourhood."""
    def __init__(self, rows, cols, dim):
        self.rows = rows
        self.cols = cols
        self.dim = dim
        self.n = rows * cols
        self.weights = np.random.randn(self.n, dim).astype(np.float32)
        for i in range(self.n):
            self.weights[i] /= np.linalg.norm(self.weights[i]) + 1e-8

    def bmu(self, x):
        dists = np.linalg.norm(self.weights - x, axis=1)
        return int(np.argmin(dists))

    def train_step(self, x, lr=0.1, sigma=2.0):
        bi = self.bmu(x)
        for i in range(self.n):
            ri, ci = divmod(i, self.cols)
            rj, cj = divmod(bi, self.cols)
            d2 = (ri - rj)**2 + (ci - cj)**2
            h = math.exp(-d2 / (2 * sigma * sigma))
            self.weights[i] += lr * h * (x - self.weights[i])
            self.weights[i] /= np.linalg.norm(self.weights[i]) + 1e-8

    def train(self, data, epochs=150, lr_start=0.1, sigma_start=2.0, progress=None):
        for epoch in range(epochs):
            lr = lr_start * (1.0 - epoch / epochs)
            sigma = max(0.5, sigma_start * (1.0 - epoch / epochs))
            for x in data:
                self.train_step(x, lr=lr, sigma=sigma)
            if progress:
                progress(epoch + 1, epochs)
