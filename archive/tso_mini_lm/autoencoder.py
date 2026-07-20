import random
import numpy as np
from collections import deque
from .hierarchy import HierarchicalGraph
from .routing import FrictionRouter

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    _DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
except ImportError:
    torch = None
    HAS_CUDA = False
    _DEVICE = None


def _to_tensor(x, device=None):
    if torch is None:
        return x
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32, device=device or _DEVICE)


def _to_numpy(x):
    if torch is None or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


class TSO_MiniLM_Autoencoder:
    """
    Auto-encodeur linéaire à apprentissage local (règle Delta).

    Anti-collapse Ph27 :
    - Bruit débruiteur sur s
    - Répulsion running_mean
    - Échantillonnage négatif contrastif (buffer circulaire)
    """

    def __init__(self, hierarchy: HierarchicalGraph, lr=5e-4, friction_threshold=0.3,
                 lambda_rep=0.01, ema_momentum=0.9, noise_std=0.0,
                 lambda_contrast=0.0, buffer_size=128):
        self.hierarchy = hierarchy
        self.router = FrictionRouter(hierarchy, friction_threshold)
        self.lr = lr
        self.lambda_rep = lambda_rep
        self.ema_momentum = ema_momentum
        self.noise_std = noise_std
        self.lambda_contrast = lambda_contrast
        self.buffer_size = buffer_size
        self.s_buffer = deque(maxlen=buffer_size)
        self.device = _DEVICE
        self._use_torch = torch is not None and self.device.type == "cuda"
        M = hierarchy.max_beta + 3

        if self._use_torch:
            self.W_enc = torch.randn(M, 384, device=self.device, dtype=torch.float32) * 0.01
            self.W_dec = torch.randn(M, 384, device=self.device, dtype=torch.float32) * 0.01
            self.running_mean = torch.zeros(384, device=self.device, dtype=torch.float32)
        else:
            self.W_enc = np.random.randn(M, 384).astype(np.float32) * 0.01
            self.W_dec = np.random.randn(M, 384).astype(np.float32) * 0.01
            self.running_mean = np.zeros(384, dtype=np.float32)

    @property
    def M(self):
        return self.hierarchy.max_beta + 3

    # ── opérations cœur (GPU si dispo) ──────────────────────────

    def encode(self, s):
        if self._use_torch and not isinstance(s, torch.Tensor):
            s = _to_tensor(s, self.device)
        return self.W_enc.T @ s

    def decode(self, h):
        if self._use_torch and not isinstance(h, torch.Tensor):
            h = _to_tensor(h, self.device)
        return self.W_dec @ h

    # ── interface phrase ────────────────────────────────────────

    def get_state(self, sentence_tokens, already_processed=False):
        self.router.phi_history.clear()
        if already_processed:
            s = self.hierarchy.build_state_from_active(sentence_tokens)
        else:
            s = self.hierarchy.get_topological_state(sentence_tokens)
        if self._use_torch:
            s = _to_tensor(s, self.device)
        return s

    def predict_embedding(self, sentence_tokens):
        s = self.get_state(sentence_tokens)
        h = self.encode(s)
        h = _to_numpy(h)
        nh = np.linalg.norm(h)
        return h / nh if nh > 1e-8 else h

    def route(self, sentence_tokens):
        surprise = getattr(self, 'last_surprise', 0.0)
        return self.router.route(sentence_tokens, surprise=surprise)

    # ── contraste local ────────────────────────────────────────

    def _contrastive_update(self, s_in, h):
        """Échantillonne un négatif du buffer et met à jour W_enc pour
        minimiser la similarité cosinus entre h et h_neg."""
        if not self.s_buffer:
            return
        # np.random.choice ne gère pas les tableaux multidim, on utilise un index
        idx = np.random.randint(0, len(self.s_buffer))
        s_neg = self.s_buffer[idx]
        if self._use_torch:
            s_neg = _to_tensor(s_neg, self.device)
            h_neg = self.encode(s_neg)
            nh = torch.norm(h)
            nh_neg = torch.norm(h_neg)
            if nh < 1e-8 or nh_neg < 1e-8:
                return
            sim = torch.dot(h / nh, h_neg / nh_neg)
            # grad_h = d/dh cos(h, h_neg) — direction qui éloigne h de h_neg
            grad_h = h_neg / (nh * nh_neg) - sim * h / (nh * nh)
            self.W_enc -= self.lr * self.lambda_contrast * torch.outer(s_in, grad_h)
        else:
            h_neg = self.encode(s_neg)
            nh = np.linalg.norm(h)
            nh_neg = np.linalg.norm(h_neg)
            if nh < 1e-8 or nh_neg < 1e-8:
                return
            sim = np.dot(h / nh, h_neg / nh_neg)
            grad_h = h_neg / (nh * nh_neg) - sim * h / (nh * nh)
            self.W_enc -= self.lr * self.lambda_contrast * np.outer(s_in, grad_h)

    # ── apprentissage local (Delta rule, sur GPU) ───────────────
    def train_step(self, sentence_tokens):
        """Apprentissage local : débruiteur + répulsion + contraste.
        Ph28 : utilise route() pour activer le recrutement β forcé par conflit."""
        # Ph28 : route() déclenche force_beta_node si conflit > seuil
        self.route(sentence_tokens)
        # route() a déjà rempli active_* / on reuse l'état traité
        s = self.get_state(sentence_tokens, already_processed=True)

        if self._use_torch:
            if self.noise_std > 0:
                noise = torch.randn_like(s) * self.noise_std
                s_in = s + noise
            else:
                s_in = s

            h = self.encode(s_in)
            s_hat = self.decode(h)
            err = s - s_hat  # débruiteur : reconstruit s (pas s_in)

            self.W_dec += self.lr * torch.outer(err, h)
            delta_h = self.W_dec.T @ err
            rep_grad = 2.0 * self.lambda_rep * (h - self.running_mean)
            self.W_enc += self.lr * torch.outer(s_in, delta_h - rep_grad)
            self.running_mean = self.ema_momentum * self.running_mean + (1 - self.ema_momentum) * h

            # Contraste local
            self._contrastive_update(s_in, h)

            self.last_surprise = float(torch.norm(h - self.running_mean))
            mse = float(torch.mean(err ** 2))
        else:
            if self.noise_std > 0:
                noise = np.random.randn(*s.shape).astype(np.float32) * self.noise_std
                s_in = s + noise
            else:
                s_in = s

            h = self.encode(s_in)
            s_hat = self.decode(h)
            err = s - s_hat

            self.W_dec += self.lr * np.outer(err, h)
            delta_h = self.W_dec.T @ err
            rep_grad = 2.0 * self.lambda_rep * (h - self.running_mean)
            self.W_enc += self.lr * np.outer(s_in, delta_h - rep_grad)
            self.running_mean = self.ema_momentum * self.running_mean + (1 - self.ema_momentum) * h

            # Contraste local
            self._contrastive_update(s_in, h)

            self.last_surprise = float(np.linalg.norm(h - self.running_mean))
            mse = float(np.mean(err ** 2))

        # Buffer : stocke s (pas s_in) pour l'échantillonnage négatif
        s_np = _to_numpy(s)
        if s_np.shape == (self.M,):
            self.s_buffer.append(s_np.copy())

        return mse

    # ── sérialisation ───────────────────────────────────────────

    def state_dict(self):
        return {
            "W_enc": _to_numpy(self.W_enc),
            "W_dec": _to_numpy(self.W_dec),
            "running_mean": _to_numpy(self.running_mean),
        }

    def load_state_dict(self, state):
        W_enc = state["W_enc"]
        W_dec = state["W_dec"]
        rm = state.get("running_mean", np.zeros(384, dtype=np.float32))
        if self._use_torch:
            self.W_enc = _to_tensor(W_enc, self.device)
            self.W_dec = _to_tensor(W_dec, self.device)
            self.running_mean = _to_tensor(rm, self.device)
        else:
            self.W_enc = W_enc.astype(np.float32)
            self.W_dec = W_dec.astype(np.float32)
            self.running_mean = rm.astype(np.float32)
