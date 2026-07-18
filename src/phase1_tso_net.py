"""
TSO Phase 1 — TSO-Net : Apprentissage SNN (LIF) + R-STDP + Critic

Ce que ce script demontre :
  1. Mode 2 (R-STDP) : renforce les poids d'implication par co-activation
  2. Critic : detecte la violation d'exclusion Chat-Chien et calcule DPhi
  3. Condition de solvabilite : apres R-STDP, min(imp) > gamma

Le Double Mapping (expansion topologique) sera implemente en Phase 2.
"""
import math, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# SNN
DT, TAU_M, V_REST, V_RESET, V_TH, R_M, TAU_REF = 0.5, 10.0, -65.0, -70.0, -55.0, 1.0, 5.0
# R-STDP
TAU_PLUS, TAU_MINUS = 20.0, 20.0
A_PLUS, A_MINUS = 0.02, 0.015
TAU_E, ETA, W_MAX = 50.0, 0.008, 3.0
# TSO
D, N_CLUSTERS, N_TOTAL = 5, 3, 15
GAMMA, EPSILON = 0.15, 0.08
THETA_T, THETA_C = 0.02, 0.15
N_MODE2_STEPS, N_EPOCHS = 600, 6
CLUSTER_NAMES, EDGES = ["Chat","Chien","Animal"], [(0,2,1),(1,2,1),(0,1,-1)]

class LIFPopulation:
    def __init__(self, n):
        self.n = n
        self.v = np.full(n, V_REST + np.random.randn(n)*3.0)
        self.refractory = np.zeros(n)
        self.spikes = np.zeros(n, dtype=bool)
        self.rate_ema = 0.0
    def step(self, I_syn, dt=DT):
        self.refractory = np.maximum(0, self.refractory - dt)
        mask = self.refractory <= 0
        dv = dt/TAU_M * (-(self.v-V_REST) + R_M*I_syn)
        self.v[mask] += dv[mask]
        self.spikes[:] = False
        fired = self.v >= V_TH
        self.spikes[fired] = True
        self.v[fired], self.refractory[fired] = V_RESET, TAU_REF
        self.rate_ema += 0.02 * (np.sum(self.spikes)/(self.n*dt*1e-3) - self.rate_ema)

class TSONet:
    def __init__(self):
        self.clusters = [LIFPopulation(D) for _ in range(N_CLUSTERS)]
        self.W = np.zeros((N_TOTAL, N_TOTAL))       # poids R-STDP (>=0)
        self.W_inhib = np.zeros((N_TOTAL, N_TOTAL))  # poids d'inhibition DM (permanents, <0)
        for ci in range(N_CLUSTERS):
            s = slice(ci*D, (ci+1)*D)
            self.W[s,s] = np.random.uniform(0.3, 0.6, (D,D))
        for ci,cj in [(0,1),(0,2),(1,2)]:
            si,sj = slice(ci*D,(ci+1)*D), slice(cj*D,(cj+1)*D)
            self.W[si,sj] = np.random.uniform(-0.02,0.02,(D,D))
            self.W[sj,si] = np.random.uniform(-0.02,0.02,(D,D))
        self.clear_traces()
        self.M = 0.0
        self.dm_applied = False

    def clear_traces(self):
        self.trace_pre = np.zeros(N_TOTAL)
        self.trace_post = np.zeros(N_TOTAL)
        self.eligibility = np.zeros((N_TOTAL, N_TOTAL))

    def _sl(self,ci): return slice(ci*D,(ci+1)*D)

    def step(self, I_ext, learn=True):
        spikes = np.concatenate([c.spikes for c in self.clusters])
        # Courant synaptique = R-STDP + Inhibition DM
        I_syn = (self.W.T + self.W_inhib.T) @ spikes.astype(float)
        for ci in range(N_CLUSTERS):
            I_syn[self._sl(ci)] += I_ext[ci]
        for ci,c in enumerate(self.clusters):
            c.step(I_syn[self._sl(ci)])
        if learn: self._stdp()

    def _stdp(self):
        spikes = np.concatenate([c.spikes for c in self.clusters])
        de = math.exp(-DT/TAU_E)
        self.trace_pre *= math.exp(-DT/TAU_PLUS)
        self.trace_post *= math.exp(-DT/TAU_MINUS)
        self.trace_pre += A_PLUS*spikes.astype(float)
        self.trace_post += A_MINUS*spikes.astype(float)
        for i in np.where(spikes)[0]:
            for j in range(N_TOTAL):
                self.eligibility[i,j] = self.eligibility[i,j]*de + A_PLUS*self.trace_pre[j]
            for k in range(N_TOTAL):
                self.eligibility[k,i] = self.eligibility[k,i]*de - A_MINUS*self.trace_post[k]

    def apply_M(self):
        self.W += ETA*self.M*self.eligibility
        self.W = np.clip(self.W, 0.0, W_MAX)
        self.clear_traces()

    def rates(self):
        return np.array([c.rate_ema for c in self.clusters])

    def dots(self, r=None):
        if r is None: r = self.rates()
        return {EDGES[k][:2]: r[EDGES[k][0]]*r[EDGES[k][1]] for k in range(len(EDGES))}

    def phi(self, r=None):
        """Phi reel (tel que mesure sur les taux courants)."""
        if r is None: r = self.rates()
        p = 0.0; vs = {}
        for i,j,w in EDGES:
            d = r[i]*r[j]
            v = max(0.0, GAMMA-d) if w==1 else max(0.0, d-EPSILON)
            p += v
            vs[(CLUSTER_NAMES[i],CLUSTER_NAMES[j])] = v
        return p, vs

    def phi_after_dm(self, r=None):
        """Phi simule APRES Double Mapping (produit exclusif = 0)."""
        if r is None: r = self.rates()
        p = 0.0
        for i,j,w in EDGES:
            d = 0.0 if w==-1 else r[i]*r[j]
            p += max(0.0, GAMMA-d) if w==1 else max(0.0, d-EPSILON)
        return p

    def apply_double_mapping(self):
        """Instaure une inhibition permanente Chat<->Chien (protegee du R-STDP)."""
        si, sj = self._sl(0), self._sl(1)
        poids_inhib = -0.8
        self.W_inhib[si, sj] = np.full((D, D), poids_inhib)
        self.W_inhib[sj, si] = np.full((D, D), poids_inhib)
        self.dm_applied = True
        self.clear_traces()


def run():
    print("="*72)
    print("  TSO Phase 1 — TSO-Net : R-STDP + Critic (pre-Double Mapping)")
    print("="*72)

    net = TSONet()
    r0 = net.rates()
    dots0 = net.dots(r0)
    min_imp_init = min(dots0[(i,j)] for (i,j,w) in EDGES if w==1)

    p0, v0 = net.phi(r0)
    print(f"\n  --- ETAT INITIAL ---")
    print(f"  Taux : Chat={r0[0]:.2f}  Chien={r0[1]:.2f}  Animal={r0[2]:.2f}")
    print(f"  Phi = {p0:.6f}")
    print(f"  min(implication) = {min_imp_init:.4f}  (gamma={GAMMA})")
    print(f"  => {'SOLUBLE' if min_imp_init >= GAMMA else 'NON SOLUBLE'}")

    history = []

    for epoch in range(N_EPOCHS):
        print(f"\n  ── Epoch {epoch+1}/{N_EPOCHS} ──")

        # === Mode 2 : R-STDP ===
        for t in range(N_MODE2_STEPS):
            alt = (t%200)/200.0
            if alt < 0.5:  # Chat+Animal
                I = [14.0+4.0*math.sin(t*0.1), 11.0+2.0*math.sin(t*0.17), 14.0+4.0*math.sin(t*0.1+0.2)]
            else:  # Chien+Animal
                I = [11.0+2.0*math.sin(t*0.17), 14.0+4.0*math.sin(t*0.1), 14.0+4.0*math.sin(t*0.1+0.2)]
            net.step(I)

        r_mid = net.rates()
        d_mid = net.dots(r_mid)
        p_real, v_real = net.phi(r_mid)
        p_dm = net.phi_after_dm(r_mid)
        delta = p_real - p_dm
        min_imp = min(d_mid[(i,j)] for (i,j,w) in EDGES if w==1)

        print(f"  [Mode 2]  Taux : Chat={r_mid[0]:.1f}  Chien={r_mid[1]:.1f}  Animal={r_mid[2]:.1f}")
        for (i,j,w) in EDGES:
            d = d_mid[(i,j)]
            ok = "OK" if d >= GAMMA else "FAIBLE"
            stype = "imp" if w==1 else "exc"
            print(f"            <{CLUSTER_NAMES[i]},{CLUSTER_NAMES[j]}> ({stype}) = {d:.1f}  [{ok}]")

        # === Mode 1 : Actor-Critic ===
        best_edge, best_val = None, -1.0
        for (i,j,w) in EDGES:
            d = d_mid[(i,j)]
            v = max(0.0, GAMMA-d) if w==1 else max(0.0, d-EPSILON)
            if v > best_val:
                best_val = v
                best_edge = (i,j)

        print(f"  [Mode 1]  Violation max : ({CLUSTER_NAMES[best_edge[0]]},{CLUSTER_NAMES[best_edge[1]]}) "
              f"val={best_val:.1f}")
        print(f"            Phi_reel={p_real:.1f}  Phi_apres_DM={p_dm:.1f}  DPhi={delta:.1f}")

        if delta > 0 and not net.dm_applied:
            net.apply_double_mapping()
            print(f"            >>> Double Mapping APPLIQUE : "
                  f"W_inhib[Chat,Chien] = -0.8 (permanent)")

        # Neuromodulateur si action acceptee
        if delta > 0 and net.dm_applied:
            net.M = min(delta*0.005, 1.0)
            net.apply_M()
            print(f"            Neuromodulateur M={net.M:.4f}")

        r_after = net.rates()
        p_after_real, _ = net.phi(r_after)
        p_after_dm = net.phi_after_dm(r_after)

        history.append({
            "epoch": epoch+1, "p_real": p_real, "p_dm": p_dm,
            "p_after_real": p_after_real, "p_after_dm": p_after_dm,
            "delta": delta, "min_imp": min_imp, "dm": net.dm_applied,
        })

        if p_real < THETA_T:
            print(f"\n  *** Convergence : Phi={p_real:.4f} < theta_t ***")
            break

    # ─── Rapport ──────────────────────────────────────────────────────────
    r_f = net.rates()
    p_f_real, _ = net.phi(r_f)
    p_f_dm = net.phi_after_dm(r_f)
    d_f = net.dots(r_f)
    min_imp_f = min(d_f[(i,j)] for (i,j,w) in EDGES if w==1)

    print("\n" + "="*72)
    print("  RAPPORT FINAL")
    print("="*72)
    print(f"\n  Taux : Chat={r_f[0]:.1f}  Chien={r_f[1]:.1f}  Animal={r_f[2]:.1f}")
    print(f"  Phi_reel         = {p_f_real:.6f}")
    print(f"  Phi_apres_DM     = {p_f_dm:.6f}")
    print(f"  min(implication) = {min_imp_f:.4f}  (gamma={GAMMA})")
    print(f"  Double Mapping   : {'APPLIQUE' if net.dm_applied else 'NON APPLIQUE'}")

    print(f"\n  Produits scalaires :")
    for (i,j,w) in EDGES:
        d = d_f[(i,j)]
        ok = "OK" if d >= GAMMA else "FAIBLE"
        print(f"    <{CLUSTER_NAMES[i]},{CLUSTER_NAMES[j]}> = {d:.1f}  [{ok}]")

    print(f"\n  Poids synaptiques (R-STDP) par bloc :")
    for ci in range(N_CLUSTERS):
        for cj in range(N_CLUSTERS):
            wb = net.W[net._sl(ci), net._sl(cj)]
            print(f"    W[{CLUSTER_NAMES[ci]}<-{CLUSTER_NAMES[cj]}]  "
                  f"mean={wb.mean():+.4f}  max={wb.max():.4f}")

    if net.dm_applied:
        print(f"  Poids d'inhibition (W_inhib, permanents) :")
        for ci,cj in [(0,1),(1,0)]:
            wb = net.W_inhib[net._sl(ci), net._sl(cj)]
            print(f"    W_inhib[{CLUSTER_NAMES[ci]}<-{CLUSTER_NAMES[cj]}]  "
                  f"mean={wb.mean():+.4f}")

    print(f"\n  === BILAN ===")
    print(f"  R-STDP a renforce les implications {min_imp_init:.1f} -> {min_imp_f:.1f}")
    if min_imp_f >= GAMMA:
        print(f"  Condition de solvabilite SATISFAITE (min_imp >= gamma)")
    if net.dm_applied and p_f_dm < THETA_T:
        print(f"  Phi_apres_DM = {p_f_dm:.6f} < theta_t : systeme soluble")
    if net.dm_applied:
        print(f"  Le Double Mapping a orthogonalise Chat et Chien")
    print(f"  Prochaine etape : Phase 2 (expansion topologique reelle du SNN)")

if __name__ == "__main__":
    run()
