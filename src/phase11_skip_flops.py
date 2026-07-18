"""
TSO Phase 11 — Skip Calcul Dynamique (FLOPs événementiels)
Démontre que la friction Φ (calculée depuis le graphe logique appris)
pilote la consommation de calcul :
  - Φ≈0 (séquence triviale)   → quasi-zero spikes parasites, FLOPs minimaux
  - Φ>0 (paradoxe contextuel) → cascade de spikes + Double Mapping, FLOPs max
"""
import numpy as np
import time, math

device = "cpu"
print("=" * 72)
print("  TSO Phase 11 — Skip Calcul Dynamique (FLOPs événementiels)")
print("=" * 72)

# ---------------------------------------------------------------------------
# Constants (identiques à Phase 3)
# ---------------------------------------------------------------------------
DT = 0.5
TAU_M = 8.0
V_REST, V_RESET, V_TH = -65.0, -66.0, -53.0
R_M = 1.0
TAU_REF = 2.0
D = 5
GAMMA, EPSILON = 0.15, 0.08
N_STEPS_PER_TOKEN = 100
N_DOUBLEMAP_STEPS = 50       # extra steps for Double Mapping resolution

# ---------------------------------------------------------------------------
# LIF Cluster
# ---------------------------------------------------------------------------
class LIFCluster:
    def __init__(self, n=D):
        self.n = n
        self.v = np.full(n, V_REST + np.random.randn(n) * 3.0)
        self.refractory = np.zeros(n)
        self.spikes = np.zeros(n, dtype=bool)
        self.rate_ema = 0.0

    def step(self, I_syn):
        self.refractory = np.maximum(0, self.refractory - DT)
        mask = self.refractory <= 0
        dv = DT / TAU_M * (-(self.v - V_REST) + R_M * I_syn)
        self.v[mask] += dv[mask]
        self.spikes[:] = False
        fired = self.v >= V_TH
        self.spikes[fired] = True
        self.v[fired], self.refractory[fired] = V_RESET, TAU_REF
        self.rate_ema += 0.02 * (np.sum(self.spikes) / max(self.n * DT * 1e-3, 1e-9) - self.rate_ema)

# ---------------------------------------------------------------------------
# TSO Core — Skip SNN
# ---------------------------------------------------------------------------
class TSONet:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.N_MAX = n_clusters * D
        self.clusters = [LIFCluster(D) for _ in range(n_clusters)]
        # weight matrix with pre-defined edges
        self.W = np.zeros((self.N_MAX, self.N_MAX))
        for ci in range(n_clusters):
            s = slice(ci * D, (ci + 1) * D)
            self.W[s, s] = np.random.uniform(0.3, 0.6, (D, D))

        self.edges = []               # (ci, cj, w)  w=+1 implication, w=-1 exclusion
        self.t = 0

        # cumulative metrics
        self.total_flops = 0.0
        self.total_spikes = 0
        self.token_log = []

    def add_edge(self, ci, cj, w):
        self.edges.append((ci, cj, w))
        s_i = slice(ci * D, (ci + 1) * D)
        s_j = slice(cj * D, (cj + 1) * D)
        boost = 0.8 if w == 1 else 0.4
        self.W[s_i, s_j] += boost
        self.W[s_j, s_i] += boost
        np.clip(self.W, 0.0, 3.0, out=self.W)

    def get_I_ext(self, sims):
        I_ext = np.maximum(0, sims) * 20.0 + 3.0
        return I_ext

    def phi(self):
        rates = np.array([c.rate_ema for c in self.clusters])
        p = 0.0
        for ci, cj, w in self.edges:
            # both clusters must be meaningfully active for edge to apply
            if rates[ci] < 0.5 or rates[cj] < 0.5:
                continue
            r = rates[ci] * rates[cj]
            if w == 1:
                p += max(0.0, GAMMA - r)
            else:
                p += max(0.0, r - EPSILON)
        return p

    def simulate(self, n_steps, I_ext, count_flops=True):
        token_spikes = 0
        for step in range(n_steps):
            all_spikes = np.concatenate([c.spikes for c in self.clusters])
            n_spikes = int(all_spikes.sum())

            I_syn = self.W.T @ all_spikes.astype(float)
            if count_flops:
                self.total_flops += n_spikes * self.N_MAX
                self.total_flops += self.N_MAX

            for ci, c in enumerate(self.clusters):
                s = slice(ci * D, (ci + 1) * D)
                c.step(I_syn[s] + I_ext[ci])
                if count_flops:
                    self.total_flops += c.n * 5

            self.t += 1
            token_spikes += n_spikes

            if n_spikes > 0 and count_flops:
                self.total_flops += n_spikes * self.N_MAX

        self.total_spikes += token_spikes
        return token_spikes

    def process_token(self, sims):
        flops_start = self.total_flops

        if flops_start == 0:
            self.total_flops += self.n_clusters * 384 * 2

        I_ext = self.get_I_ext(sims)

        spikes_base = self.simulate(N_STEPS_PER_TOKEN, I_ext, count_flops=True)

        p = self.phi()

        spikes_dm = 0
        if p > 0.01:
            spikes_dm = self.simulate(N_DOUBLEMAP_STEPS, I_ext * 0.5, count_flops=True)
            self.total_flops += self.N_MAX * self.N_MAX * 2
            spikes_dm += self.simulate(N_DOUBLEMAP_STEPS, I_ext * 0.3, count_flops=True)

        info = {
            "phi": p,
            "spikes": spikes_base + spikes_dm,
            "flops": self.total_flops - flops_start,
            "rates": [c.rate_ema for c in self.clusters],
            "sims": sims.copy(),
        }
        self.token_log.append(info)
        return info

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n  [Phase 11] Chargement de MiniLM...")
    from real_embedder import RealEmbedder
    emb = RealEmbedder()

    # centroids for our 3 concepts
    words = {"CHAT": "chat", "ANIMAL": "animal", "CHIEN": "chien"}
    centroids = {}
    for label, word in words.items():
        v = emb.embed(word)
        centroids[label] = v / np.linalg.norm(v)

    def get_sims(tok, normalize=True):
        v = emb.embed(tok)
        if normalize:
            v = v / np.linalg.norm(v)
        return np.array([float(v @ centroids[k]) for k in ["CHAT", "ANIMAL", "CHIEN"]])

    # -----------------------------------------------------------------------
    # Longer sequences for clearer contrast
    # -----------------------------------------------------------------------
    def run_sequence(label, edges, tokens):
        print(f"\n" + "-" * 72)
        print(f"  {label}")
        edge_desc = " | ".join(f"C{e[0]}→C{e[1]}({'impl' if e[2]>0 else 'excl'})" for e in edges)
        print(f"  Graphe: {edge_desc}")
        print("-" * 72)

        net = TSONet(3)
        for ci, cj, w in edges:
            net.add_edge(ci, cj, w)

        accum = {"flops": 0.0, "spikes": 0}
        for tok in tokens:
            sims = get_sims(tok)
            info = net.process_token(sims)
            accum["flops"] += info["flops"]
            accum["spikes"] += info["spikes"]
            r_str = " | ".join(f"r{c}={v:.2f}" for c, v in
                               zip(["CHAT","ANIMAL","CHIEN"], info["rates"]))
            dm = " ◈ DM" if info["phi"] > 0.01 else ""
            print(f"  [{tok:8s}] Φ={info['phi']:>7.1f}  "
                  f"spikes={info['spikes']:3d}  "
                  f"FLOPs={info['flops']:>8.0f}{dm}  {r_str}")

        print(f"  ─────────────────────────────────────────────")
        print(f"  CUMUL   spikes={accum['spikes']:3d}  FLOPs={accum['flops']:>8.0f}")
        return accum

    trivial = run_sequence(
        "SÉQUENCE TRIVIALE  : chat·est·animal·est·animal",
        [(0, 1, +1)],
        ["chat", "est", "animal", "est", "animal"],
    )

    paradox = run_sequence(
        "SÉQUENCE PARADOXALE : chat·est·chien·est·chien  (◈ = Double Mapping)",
        [(0, 2, -1)],
        ["chat", "est", "chien", "est", "chien"],
    )

    # -----------------------------------------------------------------------
    #  Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  RÉSULTATS — Efficacité Événementielle")
    print("=" * 72)
    tf, pf = trivial["flops"], paradox["flops"]
    ts, ps = trivial["spikes"], paradox["spikes"]
    print(f"  Tokens triviaux (5)  : {tf:>9.0f} FLOPs, {ts:3d} spikes")
    print(f"  Tokens paradoxaux (5): {pf:>9.0f} FLOPs, {ps:3d} spikes")
    print(f"  Ratio FLOPs          : {pf/max(tf,1):.2f}x")
    print(f"  Économie triviale     : "
          f"{100*(1-tf/pf):.0f}% de FLOPs en moins vs paradoxe")
    est_base = 9000   # min per token (embedding + BMU)
    print(f"\n  ---- Mise en perspective ----")
    print(f"  Coût fixe par token (embed+BMU): ~{est_base} FLOPs")
    print(f"  Coût SNN variable : "
          f"{tf - 5*est_base:>7.0f} FLOPs (trivial) vs "
          f"{pf - 5*est_base:>7.0f} FLOPs (paradox)")
    print(f"  Ratio SNN seul    : "
          f"{(pf - 5*est_base)/max(tf - 5*est_base, 1):.1f}x")
    print(f"  Transformer : 100% FLOPs pour tous les tokens (pas de skip)")
    print()

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"  Temps: {time.time() - t0:.1f}s")
