"""
LIF neurons and clusters — the physical substrate of TSO.

Each cluster corresponds to one conceptual node. Neurons within a cluster
share an embedding vector and fire together to signal concept activation.
"""
import numpy as np


class LIFNeuron:
    """Single leaky integrate-and-fire neuron."""
    def __init__(self):
        self.tau_m = 10.0
        self.v_rest = -65.0
        self.v_th = -55.0
        self.v_reset = -70.0
        self.v = self.v_rest
        self.refractory = 0

    def step(self, I_syn, dt=0.5):
        if self.refractory > 0:
            self.refractory -= 1
            self.spiked = False
            return 0.0
        dv = (dt / self.tau_m) * (-(self.v - self.v_rest) + I_syn)
        self.v += dv
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.refractory = 2
            self.spiked = True
            return 1.0  # spike
        self.spiked = False
        return 0.0

    def reset(self):
        self.v = self.v_rest
        self.refractory = 0
        self.spiked = False


class LIFCluster:
    """A cluster of LIF neurons acting as one conceptual unit."""
    def __init__(self, n_neurons, label="", tau_m=10.0):
        self.n = n_neurons
        self.label = label
        self.tau_m = tau_m
        self.v_rest = -65.0
        self.v_th = -55.0
        self.v_reset = -70.0
        self.v = np.full(n_neurons, self.v_rest, dtype=np.float32)
        self.spikes = np.zeros(n_neurons, dtype=np.float32)
        self.rate = 0.0
        self.tau_rate = 50.0

    def step(self, I_syn, dt=0.5):
        dv = (dt / self.tau_m) * (-(self.v - self.v_rest) + I_syn)
        self.v += dv
        self.spikes.fill(0.0)
        fired = self.v >= self.v_th
        self.spikes[fired] = 1.0
        self.v[fired] = self.v_reset

        inst_rate = float(np.mean(self.spikes))
        a = 1.0 - np.exp(-dt / self.tau_rate)
        self.rate += a * (inst_rate - self.rate)
        return inst_rate

    def reset(self):
        self.v.fill(self.v_rest)
        self.spikes.fill(0.0)
        self.rate = 0.0
