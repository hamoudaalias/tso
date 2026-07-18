"""TSO Phase 3.1 — Pipeline: Texte -> SOM -> SNN dynamique -> Expansion.
Le reseau DECOUVRE ses clusters via la SOM, apprend les implications
par R-STDP, et declenche le Double Mapping sur contradiction structurelle.
"""
import math, random
import numpy as np
from real_embedder import RealEmbedder
from native_critic import NativeCritic

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# SNN
DT, TAU_M, V_REST, V_RESET, V_TH, R_M, TAU_REF = 0.5, 10.0, -65.0, -70.0, -55.0, 1.0, 5.0
TAU_PLUS, TAU_MINUS = 20.0, 20.0
A_PLUS, A_MINUS = 0.02, 0.015
TAU_E, ETA, W_MAX = 50.0, 0.01, 3.0
D = 5
GAMMA, EPSILON = 0.15, 0.08
THETA_T = 0.02
N_MODE2_STEPS = 600
SOM_ROWS, SOM_COLS, SOM_DIM = 5, 5, 384
SOM_LR0, SOM_SIGMA0 = 0.5, 2.0
SOM_N_EPOCHS = 100
COACT_WINDOW = 15  # fenetre de co-activation en pas
COACT_THRESHOLD = 20  # nombre de co-activations necessaires


class FakeEmbedder:
    def __init__(self):
        v1 = np.random.randn(SOM_DIM); v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(SOM_DIM); v2 -= v2.dot(v1)*v1; v2 /= np.linalg.norm(v2)
        self.embeds = {"chat": v1, "chien": -v1, "animal": (v1+v2)/np.linalg.norm(v1+v2)}
    def embed(self, word):
        w = word.lower()
        if w in self.embeds:
            return self.embeds[w].copy()
        # chercher la racine
        for k in self.embeds:
            if w.startswith(k):
                return self.embeds[k].copy() + np.random.randn(SOM_DIM)*0.05
        return np.random.randn(SOM_DIM)
    def nli(self, c1, c2):
        m = {"chat":0,"chien":1,"animal":2}
        i, j = m.get(c1, -1), m.get(c2, -1)
        if i<0 or j<0: return 0
        if i==j: return 0
        if i==2 or j==2: return 1
        return -1


class MiniSOM:
    def __init__(self, rows=SOM_ROWS, cols=SOM_COLS, dim=SOM_DIM):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.weights = np.random.randn(rows*cols, dim)
        for i in range(rows*cols):
            self.weights[i] /= np.linalg.norm(self.weights[i])
        self.labels = {}

    def bmu(self, x):
        return int(np.argmin(np.linalg.norm(self.weights - x, axis=1)))

    def train_on(self, embedder, n_epochs=SOM_N_EPOCHS, concepts=None):
        if concepts is None:
            concepts = list(getattr(embedder, 'embeds', {}).keys())
        for epoch in range(n_epochs):
            lr = SOM_LR0 * (1 - epoch/n_epochs) * (1 - epoch/n_epochs) + 0.01
            sigma = SOM_SIGMA0 * (1 - epoch/n_epochs) + 0.3
            for c in concepts:
                v = embedder.embed(c) + np.random.randn(SOM_DIM)*0.05
                bmu_idx = self.bmu(v)
                for i in range(self.rows*self.cols):
                    ri, ci = divmod(i, self.cols)
                    rj, cj = divmod(bmu_idx, self.cols)
                    d2 = (ri-rj)**2 + (ci-cj)**2
                    h = math.exp(-d2/(2*sigma*sigma))
                    self.weights[i] += lr * h * (v - self.weights[i])
                    self.weights[i] /= np.linalg.norm(self.weights[i])
        for i in range(self.rows*self.cols):
            best_d, best_c = float('inf'), None
            for c in concepts:
                v = embedder.embed(c)
                d = np.linalg.norm(self.weights[i] - v)
                if d < best_d:
                    best_d, best_c = d, c
            self.labels[i] = best_c


class LIFCluster:
    def __init__(self, n, name=""):
        self.name = name
        self.n = n
        self.v = np.full(n, V_REST + np.random.randn(n)*3.0)
        self.refractory = np.zeros(n)
        self.spikes = np.zeros(n, dtype=bool)
        self.rate_ema = 0.0
    def step(self, I_syn):
        self.refractory = np.maximum(0, self.refractory - DT)
        mask = self.refractory <= 0
        dv = DT/TAU_M * (-(self.v-V_REST) + R_M*I_syn)
        self.v[mask] += dv[mask]
        self.spikes[:] = False
        fired = self.v >= V_TH
        self.spikes[fired] = True
        self.v[fired], self.refractory[fired] = V_RESET, TAU_REF
        self.rate_ema += 0.02*(np.sum(self.spikes)/(self.n*DT*1e-3)-self.rate_ema)


class DynamicTSONet:
    def __init__(self, max_clusters=SOM_ROWS*SOM_COLS):
        self.max_clusters = max_clusters
        self.N_MAX = max_clusters * D
        self.n_clusters = 0
        self.M = 0.5
        self.clusters = []
        self.label_to_ci = {}    # label -> cluster index
        self.ci_to_label = {}    # cluster index -> label
        self.W = np.zeros((self.N_MAX, self.N_MAX))
        self.trace_pre = np.zeros(self.N_MAX)
        self.trace_post = np.zeros(self.N_MAX)
        self.eligibility = np.zeros((self.N_MAX, self.N_MAX))
        self.t = 0
        self.edges = []
        self.coact_count = {}
        self.firing_history = np.zeros((max_clusters, COACT_WINDOW), dtype=bool)
        self.firing_ptr = 0

    def _sl(self, ci): return slice(ci*D, (ci+1)*D)

    def get_or_alloc(self, label):
        if label in self.label_to_ci:
            return self.label_to_ci[label]
        ci = self.n_clusters
        if ci >= self.max_clusters: return -1
        self.clusters.append(LIFCluster(D, f"C{ci}({label})"))
        self.label_to_ci[label] = ci
        self.ci_to_label[ci] = label
        self.n_clusters += 1
        s = self._sl(ci)
        self.W[s,s] = np.random.uniform(0.3, 0.6, (D,D))
        return ci

    def all_spikes(self):
        if not self.clusters: return np.zeros(self.N_MAX)
        parts = [c.spikes for c in self.clusters]
        n_remain = self.N_MAX - self.n_clusters*D
        if n_remain > 0:
            parts.append(np.zeros(n_remain, dtype=bool))
        return np.concatenate(parts)

    def step(self, I_ext, learn=True):
        spikes = self.all_spikes()
        I_syn = self.W.T @ spikes.astype(float)
        for ci in range(self.n_clusters):
            I_syn[self._sl(ci)] += I_ext[ci]
        for ci, c in enumerate(self.clusters):
            c.step(I_syn[self._sl(ci)])
        self.t += 1
        if learn:
            self._stdp(spikes)
        # Mettre a jour l'historique de firing (fenetre glissante)
        for ci in range(self.n_clusters):
            self.firing_history[ci, self.firing_ptr] = self.clusters[ci].spikes.any()
        self.firing_ptr = (self.firing_ptr + 1) % COACT_WINDOW
        if learn:
            self._hebbian()
        # Detecter les co-activations dans la fenetre
        for ci in range(self.n_clusters):
            for cj in range(ci+1, self.n_clusters):
                if np.any(self.firing_history[ci]) and np.any(self.firing_history[cj]):
                    self.add_coact(ci, cj)


    def _stdp(self, spikes):
        de = math.exp(-DT/TAU_E)
        self.trace_pre *= math.exp(-DT/TAU_PLUS)
        self.trace_post *= math.exp(-DT/TAU_MINUS)
        s = spikes.astype(float)
        self.trace_pre += A_PLUS*s
        self.trace_post += A_MINUS*s
        for i in np.where(spikes)[0]:
            for j in range(self.N_MAX):
                self.eligibility[i,j] = self.eligibility[i,j]*de + A_PLUS*self.trace_pre[j]
            for k in range(self.N_MAX):
                self.eligibility[k,i] = self.eligibility[k,i]*de - A_MINUS*self.trace_post[k]

    def _hebbian(self):
        """Hebbien sur la fenetre de co-activation : renforce les paires co-actives."""
        for ci in range(self.n_clusters):
            if not np.any(self.firing_history[ci]):
                continue
            s_i = self._sl(ci)
            for cj in range(ci + 1, self.n_clusters):
                if not np.any(self.firing_history[cj]):
                    continue
                s_j = self._sl(cj)
                self.W[s_i, s_j] += 0.015
                self.W[s_j, s_i] += 0.015
        self.W = np.clip(self.W, 0.0, W_MAX)

    def apply_M(self):
        self.W += ETA*self.M*self.eligibility
        self.W = np.clip(self.W, 0.0, W_MAX)
        self.trace_pre.fill(0.0); self.trace_post.fill(0.0); self.eligibility.fill(0.0)

    def rates(self): return np.array([c.rate_ema for c in self.clusters])

    def phi(self):
        r = self.rates()
        p = 0.0
        for ci,cj,w in self.edges:
            d = r[ci]*r[cj] if ci<len(r) and cj<len(r) else 0.0
            p += max(0.0, GAMMA-d) if w==1 else max(0.0, d-EPSILON)
        return p

    def min_imp(self):
        r = self.rates()
        vals = [r[ci]*r[cj] for ci,cj,w in self.edges if w==1 and ci<len(r) and cj<len(r)]
        return min(vals) if vals else 0.0

    def add_coact(self, ci, cj):
        if ci!=cj: self.coact_count[(min(ci,cj),max(ci,cj))] = self.coact_count.get((min(ci,cj),max(ci,cj)),0)+1

    def finalize_edges(self, critic=None, threshold=COACT_THRESHOLD):
        """Construit le graphe via le NativeCritic (dynamique SNN interne)."""
        self.edges = []
        for (ci,cj), cnt in self.coact_count.items():
            if cnt < threshold: continue
            w = critic.evaluate(ci, cj) if critic else 0
            if w != 0:
                self.edges.append((ci,cj,w))
        n = self.n_clusters
        for ci in range(n):
            for cj in range(ci+1, n):
                if any((a==ci and b==cj) or (a==cj and b==ci) for a,b,_ in self.edges):
                    continue
                w = critic.evaluate(ci, cj) if critic else 0
                if w != 0:
                    self.edges.append((ci,cj,w))

    def merge_by_label(self):
        """Fusionner les clusters de meme label (garder le premier)."""
        keep = {}  # label -> premier ci
        for ci in range(self.n_clusters):
            lbl = self.ci_to_label.get(ci, "?")
            if lbl not in keep:
                keep[lbl] = ci
        # Si tous les labels sont deja uniques, rien a faire
        if len(keep) == self.n_clusters:
            return
        # Creer le nouveau mapping
        old_to_new = {}
        new_clusters = []
        new_W = np.zeros((self.N_MAX, self.N_MAX))
        for new_ci, lbl in enumerate(keep):
            old_ci = keep[lbl]
            old_to_new[old_ci] = new_ci
            new_clusters.append(self.clusters[old_ci])
            new_clusters[-1].name = f"C{new_ci}({lbl})"
            # Copier les poids
            s_old, s_new = self._sl(old_ci), self._sl(new_ci)
            new_W[s_new] = self.W[s_old].copy()
        self.clusters = new_clusters
        self.W = new_W
        self.n_clusters = len(keep)
        self.label_to_ci = {lbl: i for i, lbl in enumerate(keep)}
        self.ci_to_label = {i: lbl for i, lbl in enumerate(keep)}
        # Re-indexer le graphe
        self.edges = [(old_to_new[ci], old_to_new[cj], w) for ci,cj,w in self.edges if ci in old_to_new and cj in old_to_new]
        # Re-indexer les co-activations
        self.coact_count = {(old_to_new[ci], old_to_new[cj]): cnt
                           for (ci,cj), cnt in self.coact_count.items()
                           if ci in old_to_new and cj in old_to_new}


def trigger_dm(net, embedder):
    """Double Mapping : cree les clusters C2 pour les exclusifs + Animal_C2."""
    exc_pairs = [(ci,cj) for ci,cj,w in net.edges if w == -1]
    if not exc_pairs:
        return
    # Contexte commun = cluster implique par les deux exclusifs
    imp_to = {}
    for ci,cj,w in net.edges:
        if w == 1:
            imp_to.setdefault(ci, set()).add(cj)
            imp_to.setdefault(cj, set()).add(ci)
    ctx_candidates = [c for c,prems in imp_to.items() if len(prems) >= 2]
    if not ctx_candidates:
        return
    ctx = ctx_candidates[0]
    exclusifs = list(imp_to[ctx])
    ctx_lbl = net.ci_to_label.get(ctx, "?")

    exc_strs = [f"C{e}({net.ci_to_label.get(e,'?')})" for e in exclusifs]
    print(f"\n      >>> DOUBLE MAPPING: contexte=C{ctx}({ctx_lbl}), "
          f"exclusifs={exc_strs}")

    # 1) Creer Animal_C2 (pont de contexte)
    animal_c2_label = f"{ctx_lbl}_C2"
    ctx2 = net.get_or_alloc(animal_c2_label)
    if ctx2 < 0:
        print("      ERREUR: impossible d'allouer Animal_C2")
        return

    # Copier les poids d'Animal_C1 vers Animal_C2
    s_ctx = net._sl(ctx)
    s_ctx2 = net._sl(ctx2)
    net.W[s_ctx2, s_ctx2] = np.random.uniform(0.3, 0.6, (D,D))
    # Pont bidirectionnel Animal_C1 <-> Animal_C2
    net.W[s_ctx, s_ctx2] = net.W[s_ctx, s_ctx].copy() * 0.7
    net.W[s_ctx2, s_ctx] = net.W[s_ctx, s_ctx].copy() * 0.7

    # 2) Pour chaque exclusif, creer un cluster C2
    #    et y copier les poids d'implication
    for e in exclusifs:
        lbl = net.ci_to_label.get(e, "?")
        label_c2 = f"{lbl}_C2"
        if label_c2 in net.label_to_ci:
            continue
        c2 = net.get_or_alloc(label_c2)
        if c2 < 0: continue
        se = net._sl(e)
        sc2 = net._sl(c2)
        # Copier exclusif->contexte vers C2->contexte
        net.W[sc2, s_ctx] = net.W[se, s_ctx].copy()
        net.W[s_ctx, sc2] = net.W[s_ctx, se].copy()
        # Connexion C2 <-> Animal_C2
        net.W[sc2, s_ctx2] = net.W[se, s_ctx].copy() * 0.7
        net.W[s_ctx2, sc2] = net.W[s_ctx, se].copy() * 0.7
        net.W[sc2, sc2] = np.random.uniform(0.3, 0.6, (D,D))

    # Les aretes apres DM : implications entre C2 (exclusifs) et Animal, plus le pont
    net.edges = []
    for e in exclusifs:
        lbl = net.ci_to_label.get(e, "?")
        label_c2 = f"{lbl}_C2"
        if label_c2 in net.label_to_ci:
            c2 = net.label_to_ci[label_c2]
            net.edges.append((c2, ctx, 1))  # exclusif_C2 -> Animal
        else:
            net.edges.append((e, ctx, 1))   # fallback sur le C1 original
    net.edges.append((ctx, ctx2, 1))  # pont Animal
    net.trace_pre.fill(0.0); net.trace_post.fill(0.0); net.eligibility.fill(0.0)
    net.apply_M()
    new_lbls = [net.ci_to_label.get(nc,"?") for nc in range(net.n_clusters)]
    print(f"      Nouveaux clusters: {new_lbls}")
    print(f"      Nouvelles aretes: {net.edges}")


def run_phase3():
    print("="*72)
    print("  TSO Phase 3.1 — Pipeline texte -> SOM -> SNN -> Expansion")
    print("="*72)

    embedder = RealEmbedder()
    som = MiniSOM()
    net = DynamicTSONet()
    concepts = ["cat", "dog", "animal"]

    # Phase A: entrainer la SOM
    print(f"\n  Phase A: Apprentissage SOM ({SOM_N_EPOCHS} epochs)")
    som.train_on(embedder, SOM_N_EPOCHS, concepts)
    print("  Carte SOM (concept dominant):")
    for r in range(SOM_ROWS):
        row = [som.labels[r*SOM_COLS+c][:5] for c in range(SOM_COLS)]
        print("    " + " ".join(f"{x:>5}" for x in row))

    # Phase B: Mode 2 - Consolidation R-STDP
    # Co-activation guidee : Chat+Animal, puis Chien+Animal (mode Phase 1)
    print(f"\n  Phase B: Mode 2 Consolidation ({N_MODE2_STEPS} pas)")
    for t in range(N_MODE2_STEPS):
        alt = t % 200
        if alt < 100:
            target_a, target_b = "cat", "animal"
        else:
            target_a, target_b = "dog", "animal"

        # Activer les clusters via le SOM
        va = embedder.embed(target_a) + np.random.randn(SOM_DIM)*0.03
        vb = embedder.embed(target_b) + np.random.randn(SOM_DIM)*0.03
        sia = som.bmu(va); la = som.labels.get(sia, "?")
        sib = som.bmu(vb); lb = som.labels.get(sib, "?")
        cia = net.get_or_alloc(la)
        cib = net.get_or_alloc(lb)

        I_ext = np.zeros(net.n_clusters)
        if cia >= 0: I_ext[cia] = 14.0 + 4.0*math.sin(t*0.1)
        if cib >= 0 and cib != cia: I_ext[cib] = 12.0 + 3.0*math.sin(t*0.1+0.2)
        net.step(I_ext)

    net.apply_M()
    net.merge_by_label()
    critic = NativeCritic()
    critic.attach(net)
    net.finalize_edges(critic)

    print(f"\n  Etat apres consolidation:")
    r = net.rates()
    for i in range(net.n_clusters):
        print(f"    C{i}({net.ci_to_label.get(i,'?'):>6s}) : {r[i]:.1f} Hz")
    for ci,cj,w in net.edges:
        t = "imp" if w==1 else "exc"
        d = r[ci]*r[cj] if ci<len(r) and cj<len(r) else 0
        ok = "OK" if (d>=GAMMA and w==1) or (d<=EPSILON and w==-1) else "V"
        print(f"    <C{ci},{cj}> ({t}) = {d:.1f}  [{ok}]")
    print(f"    Phi = {net.phi():.4f}  min_imp = {net.min_imp():.4f}")

    # Phase C: Critic -> Expansion
    print(f"\n  Phase C: Critic -> Expansion")
    min_imp_val = net.min_imp()
    has_exc = any(w==-1 for _,_,w in net.edges)
    print(f"    min_imp = {min_imp_val:.4f}  (gamma = {GAMMA})")
    print(f"    Contradiction detectee: {has_exc}")

    if min_imp_val >= GAMMA and has_exc:
        trigger_dm(net, embedder)

        # Phase D: Post-expansion - routage adapte apres DM
        # Concept exclusif     -> active le _C2 (pas le C1)
        # Animal (contexte)    -> active Animal_C1 ET Animal_C2
        print(f"\n  Phase D: Post-expansion ({N_MODE2_STEPS} pas)")
        for t in range(N_MODE2_STEPS):
            alt = t % 200
            if alt < 100:
                target_a, target_b = "cat", "animal"
            else:
                target_a, target_b = "dog", "animal"

            I_ext = np.zeros(net.n_clusters)

            # Cibler le cluster du concept exclusif (preferer _C2 si disponible)
            la_c2 = f"{target_a}_C2"
            if la_c2 in net.label_to_ci:
                cia = net.label_to_ci[la_c2]
            else:
                va = embedder.embed(target_a) + np.random.randn(SOM_DIM)*0.03
                sia = som.bmu(va)
                la = som.labels.get(sia, "?")
                cia = net.get_or_alloc(la)
            if cia >= 0: I_ext[cia] = 14.0 + 4.0*math.sin(t*0.1)

            # Cibler Animal (les deux couches)
            animal_c2_label = "animal_C2"
            if animal_c2_label in net.label_to_ci:
                cib = net.label_to_ci["animal"]    # Animal_C1 (original)
                ci_animal2 = net.label_to_ci[animal_c2_label]  # Animal_C2
                if cib >= 0: I_ext[cib] = 12.0 + 3.0*math.sin(t*0.1+0.2)
                I_ext[ci_animal2] = 12.0 + 3.0*math.sin(t*0.1+0.2)
            else:
                vb = embedder.embed(target_b) + np.random.randn(SOM_DIM)*0.03
                sib = som.bmu(vb)
                lb = som.labels.get(sib, "?")
                cib = net.get_or_alloc(lb)
                if cib >= 0 and cib != cia: I_ext[cib] = 12.0 + 3.0*math.sin(t*0.1+0.2)

            net.step(I_ext)

        net.apply_M()

        print(f"\n  Etat final post-expansion:")
        r = net.rates()
        for i in range(net.n_clusters):
            lbl = net.ci_to_label.get(i,"?")
            print(f"    C{i}({lbl:>8s}) : {r[i]:.1f} Hz")
        for ci,cj,w in net.edges:
            t_ = "imp" if w==1 else "exc"
            d = r[ci]*r[cj] if ci<len(r) and cj<len(r) else 0
            ok = "OK" if (d>=GAMMA and w==1) or (d<=EPSILON and w==-1) else "V"
            print(f"    <C{ci},{cj}> ({t_}) = {d:.1f}  [{ok}]")
        print(f"    Phi = {net.phi():.4f}")

        if net.phi() < THETA_T:
            print(f"\n  *** PHASE 3.1 VALIDEE ***")
        else:
            print(f"\n  Convergence partielle (Phi = {net.phi():.4f})")
    else:
        print("  Expansion non declenchee (condition non atteinte ou pas de contradiction)")


if __name__ == "__main__":
    run_phase3()
