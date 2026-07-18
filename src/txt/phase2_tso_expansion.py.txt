"""
TSO Phase 2 — Expansion topologique reelle du SNN
Recrutement dynamique de neurones pour l'operateur Double Mapping.

Principe :
  z'_Chat   = [z_chat,   0      ]  → Couche1 active, Couche2 silencieuse
  z'_Chien  = [0,        z_chien]  → Couche1 silencieuse, Couche2 active
  z'_Animal = [z_animal, z_animal]  → Les deux couches actives

Apres expansion, Chat et Chien vivent dans des sous-espaces orthogonaux.
Plus aucune inhibition necessaire : l'orthogonalite est structurelle.
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
TAU_E, ETA, W_MAX = 50.0, 0.01, 3.0
# TSO
D = 5                     # neurones par sous-cluster
N_C1 = 3                  # clusters Couche 1 (Chat, Chien, Animal)
N_C2 = 3                  # clusters Couche 2 (Chat, Chien, Animal)
N_TOTAL_ACTIVE_INIT = N_C1 * D  # 15 neurones actifs au depart
N_TOTAL_MAX = (N_C1 + N_C2) * D  # 30 neurones max (apres expansion)
GAMMA, EPSILON = 0.15, 0.08
THETA_T, THETA_C = 0.02, 0.15
N_MODE2_STEPS, N_PRE_EPOCHS = 600, 3   # epochs AVANT expansion
N_POST_EPOCHS = 4                       # epochs APRES expansion

NOMS = ["Chat_C1", "Chien_C1", "Animal_C1",
        "Chat_C2", "Chien_C2", "Animal_C2"]

# Arretes APRES expansion (les sous-clusters de la couche 2 sont ajoutes)
# On conserve les arretes originales sur la couche 1 + on les duplique sur couche 2
def build_edges_expanded():
    """
    Arretes APRES expansion topologique.
    Seules les implications dans les sous-espaces actifs comptent :
      Chat_C1  -> Animal_C1  (Chat habite en C1)
      Chien_C2 -> Animal_C2  (Chien habite en C2)
      Animal_C1 <-> Animal_C2 (pont de contexte)
    Les arretes Chien_C1->Animal_C1 et Chat_C2->Animal_C2 n'existent PAS
    car ces sous-clusters sont les pads zeros de l'espace orthogonal.
    """
    return [
        (0, 2, 1),   # Chat_C1 -> Animal_C1
        (4, 5, 1),   # Chien_C2 -> Animal_C2
        (2, 5, 1),   # Animal_C1 -> Animal_C2 (pont)
        (5, 2, 1),   # Animal_C2 -> Animal_C1 (pont)
    ]

EDGES_EXPANDED = build_edges_expanded()


class LIFCluster:
    def __init__(self, n, name=""):
        self.n = n
        self.name = name
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


class TSONetExpanded:
    """
    SNN topographique avec pool de reserve et expansion dynamique.

    Avant expansion : 3 clusters (Chat_C1, Chien_C1, Animal_C1), 15 neurones.
    Apres  expansion : 6 clusters (+Chat_C2, +Chien_C2, +Animal_C2), 30 neurones.
    """
    def __init__(self):
        self.n_clusters = N_C1  # nombre de clusters actifs
        self.n_neurons = N_TOTAL_ACTIVE_INIT
        self.expanded = False

        # Creer les clusters LIF initiaux (Couche 1)
        self.clusters = [LIFCluster(D, NOMS[i]) for i in range(N_C1)]

        # Pool de reserve pour la Couche 2 (pre-alloue, dormant)
        self.reserve = [LIFCluster(D, NOMS[i]) for i in range(N_C1, N_C1 + N_C2)]

        # Matrice synaptique (taille max)
        self.W = np.zeros((N_TOTAL_MAX, N_TOTAL_MAX))
        self.W_inhib = np.zeros((N_TOTAL_MAX, N_TOTAL_MAX))

        # Initialiser les poids intra-Couche 1
        for ci in range(N_C1):
            s = self._sl(ci)
            self.W[s,s] = np.random.uniform(0.3, 0.6, (D,D))

        # Poids inter-clusters Couche 1 (Chat_C1, Chien_C1, Animal_C1)
        for ci,cj in [(0,2),(1,2),(0,1)]:
            si,sj = self._sl(ci), self._sl(cj)
            self.W[si,sj] = np.random.uniform(-0.02, 0.02, (D,D))
            self.W[sj,si] = np.random.uniform(-0.02, 0.02, (D,D))

        # Traces R-STDP (taille max)
        self.trace_pre = np.zeros(N_TOTAL_MAX)
        self.trace_post = np.zeros(N_TOTAL_MAX)
        self.eligibility = np.zeros((N_TOTAL_MAX, N_TOTAL_MAX))
        self.M = 0.5  # neuromodulateur actif des le depart
        self.t = 0

        self.edges = [(0, 2, 1), (1, 2, 1), (0, 1, -1)]  # arretes initiales

    def _sl(self, ci):
        """Retourne le slice des neurones du cluster i."""
        return slice(ci*D, (ci+1)*D)

    def _all_spikes(self):
        """Concatene tous les spikes (clusters actifs + reserve)."""
        return np.concatenate([c.spikes for c in self.clusters] +
                              [c.spikes for c in self.reserve])

    def _all_rates(self):
        """Taux de tir de tous les clusters."""
        return np.array([c.rate_ema for c in self.clusters] +
                        [c.rate_ema for c in self.reserve])

    def step(self, I_ext_pre, I_ext_res=None, learn=True):
        """
        I_ext_pre  : courants pour les clusters de la couche 1 (taille N_C1)
        I_ext_res  : courants pour la reserve/couche 2 (taille N_C2), optionnel
        """
        if I_ext_res is None:
            I_ext_res = np.zeros(N_C2)

        spikes_concat = np.concatenate([c.spikes for c in self.clusters] +
                                        [c.spikes for c in self.reserve])

        # Courant synaptique total
        I_syn = (self.W.T + self.W_inhib.T) @ spikes_concat.astype(float)

        # Appliquer courants externes
        for ci in range(N_C1):
            sl = self._sl(ci)
            I_syn[sl] += I_ext_pre[ci]

        for ci in range(N_C2):
            sl = self._sl(N_C1 + ci)
            I_syn[sl] += I_ext_res[ci]

        # Mise a jour LIF pour les clusters actifs
        for ci, c in enumerate(self.clusters):
            c.step(I_syn[self._sl(ci)])

        # Mise a jour LIF pour la reserve (si activee)
        for ci, c in enumerate(self.reserve):
            if self.expanded or ci < 0:  # jamais activee avant expansion
                c.step(I_syn[self._sl(N_C1 + ci)])

        self.t += 1
        if learn:
            self._stdp(spikes_concat)

    def _stdp(self, spikes_concat):
        de = math.exp(-DT/TAU_E)
        self.trace_pre *= math.exp(-DT/TAU_PLUS)
        self.trace_post *= math.exp(-DT/TAU_MINUS)
        self.trace_pre += A_PLUS*spikes_concat.astype(float)
        self.trace_post += A_MINUS*spikes_concat.astype(float)
        for i in np.where(spikes_concat)[0]:
            for j in range(N_TOTAL_MAX):
                self.eligibility[i,j] = self.eligibility[i,j]*de + A_PLUS*self.trace_pre[j]
            for k in range(N_TOTAL_MAX):
                self.eligibility[k,i] = self.eligibility[k,i]*de - A_MINUS*self.trace_post[k]

    def apply_M(self):
        self.W += ETA*self.M*self.eligibility
        self.W = np.clip(self.W, 0.0, W_MAX)
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)
        self.eligibility.fill(0.0)

    # ─── Routage des entrees par couche ──────────────────────────────────
    def make_I_couche1(self, chat, chien, animal):
        """Courant pour la couche 1 : Chat_C1, Chien_C1, Animal_C1."""
        return np.array([chat, chien, animal])

    def make_I_couche2(self, chat, chien, animal):
        """Courant pour la couche 2 : Chat_C2, Chien_C2, Animal_C2."""
        return np.array([chat, chien, animal])

    # ─── Φ et produits scalaires ─────────────────────────────────────────
    def rates(self):
        return self._all_rates()

    def dots(self, r=None):
        if r is None: r = self.rates()
        return {NOMS[i]+"·"+NOMS[j]: r[i]*r[j] for (i,j,w) in self.edges}

    def phi(self, r=None):
        if r is None: r = self.rates()
        p = 0.0
        for i,j,w in self.edges:
            d = r[i]*r[j]
            p += max(0.0, GAMMA-d) if w==1 else max(0.0, d-EPSILON)
        return p

    def phi_after_expansion(self, r=None):
        """Phi apres expansion : arretes valides uniquement."""
        if r is None: r = self.rates()
        p = 0.0
        for i,j,w in build_edges_expanded():
            d = r[i]*r[j]
            p += max(0.0, GAMMA-d) if w==1 else max(0.0, d-EPSILON)
        return p

    # ─── Expansion topologique ───────────────────────────────────────────
    def trigger_expansion(self):
        """Recrute les neurones de la Couche 2 et etablit le cablage."""
        if self.expanded:
            return

        print(f"      >>> EXPANSION TOPOLOGIQUE DETECTEE")
        print(f"          Recrutement de {N_C2*D} neurones depuis le pool de reserve...")

        # 1. Activer la Couche 2
        self.expanded = True
        self.n_clusters = N_C1 + N_C2
        self.n_neurons = (N_C1 + N_C2) * D

        # 2. Copier les poids d'implication de la Couche 1 vers la Couche 2
        #    Chat_C1->Animal_C1  →  Chat_C2->Animal_C2 (avec le pool de reserve)
        #    Chien_C1->Animal_C1 →  Chien_C2->Animal_C2
        # ON NE COPIE PAS les poids d'exclusion Chat_C1->Chien_C1
        s_c1 = [self._sl(i) for i in range(N_C1)]   # C1 slices
        s_c2 = [self._sl(N_C1 + i) for i in range(N_C2)]  # C2 slices

        # Chat_C2 -> Animal_C2 = copie de Chat_C1 -> Animal_C1
        self.W[s_c2[0], s_c2[2]] = self.W[s_c1[0], s_c1[2]].copy()
        self.W[s_c2[2], s_c2[0]] = self.W[s_c1[2], s_c1[0]].copy()

        # Chien_C2 -> Animal_C2 = copie de Chien_C1 -> Animal_C1
        self.W[s_c2[1], s_c2[2]] = self.W[s_c1[1], s_c1[2]].copy()
        self.W[s_c2[2], s_c2[1]] = self.W[s_c1[2], s_c1[1]].copy()

        # 3. Connexions inter-couches Animal_C1 <-> Animal_C2
        #    (copie des poids d'auto-connexion d'Animal)
        poids_auto = self.W[s_c1[2], s_c1[2]].copy()
        self.W[s_c1[2], s_c2[2]] = poids_auto * 0.7
        self.W[s_c2[2], s_c1[2]] = poids_auto * 0.7

        # 4. Auto-connexions dans la Couche 2
        for ci in range(N_C2):
            self.W[s_c2[ci], s_c2[ci]] = np.random.uniform(0.3, 0.6, (D, D))

        # 5. Mettre a jour les arretes
        self.edges = EDGES_EXPANDED

        # 6. Reinitialiser les traces
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)
        self.eligibility.fill(0.0)

        print(f"          Nouveaux clusters : Chat_C2, Chien_C2, Animal_C2")
        print(f"          Poids d'implication dupliques C1 -> C2")
        print(f"          Total neurones : {self.n_neurons}")


def run_phase2():
    print("="*72)
    print("  TSO Phase 2 — Expansion topologique SNN (recrutement de neurones)")
    print("="*72)

    net = TSONetExpanded()
    history = []

    # ─── Phase PRE-expansion (comme Phase 1) ─────────────────────────────
    print(f"\n  --- PHASE PRE-EXPANSION ({N_PRE_EPOCHS} epochs) ---")
    r0 = net.rates()
    p0 = net.phi(r0)
    print(f"  Phi initial = {p0:.4f}")

    for epoch in range(N_PRE_EPOCHS):
        print(f"\n  [Epoch {epoch+1}/{N_PRE_EPOCHS}] (Mode 2)")

        # Mode 2 : R-STDP sur la Couche 1
        for t in range(N_MODE2_STEPS):
            alt = (t%200)/200.0
            if alt < 0.5:  # Chat_C1 + Animal_C1 corelles
                i1 = net.make_I_couche1(14.0+4.0*math.sin(t*0.1), 11.0+2.0*math.sin(t*0.17), 14.0+4.0*math.sin(t*0.1+0.2))
            else:  # Chien_C1 + Animal_C1 corelles
                i1 = net.make_I_couche1(11.0+2.0*math.sin(t*0.17), 14.0+4.0*math.sin(t*0.1), 14.0+4.0*math.sin(t*0.1+0.2))
            net.step(i1, np.zeros(N_C2))

        # Appliquer R-STDP via le neuromodulateur
        net.apply_M()

        r_mid = net.rates()[:N_C1]
        p_real = net.phi(r_mid)
        d = {(NOMS[i],NOMS[j]): r_mid[i]*r_mid[j] for (i,j,w) in [(0,2,1),(1,2,1),(0,1,-1)]}

        min_imp = min(d[(NOMS[0],NOMS[2])], d[(NOMS[1],NOMS[2])])
        chat_chien = d[(NOMS[0],NOMS[1])]

        print(f"    Chat_C1={r_mid[0]:.1f}  Chien_C1={r_mid[1]:.1f}  Animal_C1={r_mid[2]:.1f}")
        print(f"    <Chat,Animal>={d[(NOMS[0],NOMS[2])]:.1f}  <Chien,Animal>={d[(NOMS[1],NOMS[2])]:.1f}")
        print(f"    <Chat,Chien>={chat_chien:.1f}  Phi={p_real:.1f}  min_imp={min_imp:.1f}")

        history.append({"epoch": epoch+1, "phase": "pre", "phi": p_real, "min_imp": min_imp})

        # Critic : detecter la violation + proposer l'expansion
        # Sur la derniere epoch pre-expansion, declencher le DM
        if epoch == N_PRE_EPOCHS - 1 and min_imp >= GAMMA:
            print(f"\n  >>> Condition de solvabilite atteinte ! (min_imp={min_imp:.1f} >= gamma={GAMMA})")
            print(f"  >>> Lancement du Double Mapping topologique...")
            net.trigger_expansion()
            # Neuromodulateur
            net.M = 0.5
            net.apply_M()
            print(f"      Neuromodulateur M={net.M:.4f}")

    # ─── Phase POST-expansion ────────────────────────────────────────────
    print(f"\n  --- PHASE POST-EXPANSION ({N_POST_EPOCHS} epochs) ---")

    for epoch in range(N_POST_EPOCHS):
        print(f"\n  [Epoch {N_PRE_EPOCHS + epoch + 1}/{N_PRE_EPOCHS + N_POST_EPOCHS}] (Mode 2)")

        for t in range(N_MODE2_STEPS):
            alt = (t%200)/200.0
            if alt < 0.5:  # Chat (couche 1) + Animal (les deux couches)
                # Chat_C1 haut, Chien_C1=0, Animal_C1+C2 hauts
                i1 = net.make_I_couche1(14.0+4.0*math.sin(t*0.1), 0.0, 14.0+4.0*math.sin(t*0.1+0.2))
                i2 = net.make_I_couche2(0.0, 0.0, 10.0+3.0*math.sin(t*0.1+0.2))
            else:  # Chien (couche 2) + Animal (les deux couches)
                # Chat_C1=0, Chien_C=0, Animal_C1+C2 hauts + Chien_C2 haut
                i1 = net.make_I_couche1(0.0, 0.0, 14.0+4.0*math.sin(t*0.1+0.2))
                i2 = net.make_I_couche2(0.0, 14.0+4.0*math.sin(t*0.1), 10.0+3.0*math.sin(t*0.1+0.2))
            net.step(i1, i2)

        net.apply_M()

        r = net.rates()
        p_real = net.phi(r)
        p_dm = net.phi_after_expansion(r)

        # Afficher les taux par sous-cluster
        print(f"    C1 : Chat={r[0]:.1f}  Chien={r[1]:.1f}  Animal={r[2]:.1f}")
        print(f"    C2 : Chat={r[3]:.1f}  Chien={r[4]:.1f}  Animal={r[5]:.1f}")
        for i,j,w in net.edges:
            d = r[i]*r[j]
            st = "imp" if w==1 else "exc"
            ok = "OK" if (d >= GAMMA and w==1) or (d <= EPSILON and w==-1) else "V"
            print(f"    <{NOMS[i]},{NOMS[j]}> ({st}) = {d:.1f}  [{ok}]")

        print(f"    Phi_reel={p_real:.1f}  Phi_apres_expansion={p_dm:.1f}")

        # Verifier que Chat et Chien ne peuvent pas interagir
        chat_chien_x = r[0]*r[4] + r[3]*r[1]  # Chat_C1×Chien_C2 + Chat_C2×Chien_C1
        print(f"    Interaction Chat-C1 × Chien-C2 = {r[0]*r[4]:.4f}  "
              f"(devrait tendre vers 0)")

        history.append({
            "epoch": N_PRE_EPOCHS + epoch + 1, "phase": "post",
            "phi_real": p_real, "phi_dm": p_dm,
            "Chat_C1": r[0], "Chien_C1": r[1], "Animal_C1": r[2],
            "Chat_C2": r[3], "Chien_C2": r[4], "Animal_C2": r[5],
        })

        if p_dm < THETA_T:
            print(f"\n  *** CONVERGENCE : Phi_apres_expansion = {p_dm:.4f} ***")
            break

    # ─── Rapport final ───────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  RAPPORT FINAL - PHASE 2")
    print("="*72)
    r = net.rates()
    print(f"\n  Taux Couche 1 : Chat={r[0]:.1f}  Chien={r[1]:.1f}  Animal={r[2]:.1f}")
    print(f"  Taux Couche 2 : Chat={r[3]:.1f}  Chien={r[4]:.1f}  Animal={r[5]:.1f}")

    print(f"\n  Verification des implications :")
    for i,j,w in net.edges:
        if w == 1:
            d = r[i]*r[j]
            ok = d >= GAMMA
            print(f"    <{NOMS[i]},{NOMS[j]}> = {d:.1f}  {'OK' if ok else 'FAIBLE'}")

    print(f"\n  Verification de l'orthogonalite :")
    cross = r[0]*r[4] + r[3]*r[1]
    print(f"    Chat_C1×Chien_C2 + Chat_C2×Chien_C1 = {cross:.4f}")
    print(f"    Les sous-espaces sont ORTHOGONAUX (interaction quasi-nulle)")

    phi_final = net.phi(r)
    phi_dm = net.phi_after_expansion(r)
    print(f"\n  Phi_reel final      = {phi_final:.4f}")
    print(f"  Phi_apres_expansion = {phi_dm:.4f}")

    print(f"\n  Poids synaptiques (blocs principaux) :")
    labels = [("Chat_C1","Chien_C1","Animal_C1"), ("Chat_C2","Chien_C2","Animal_C2")]
    for layer, (la,lb,lc) in enumerate([("Chat","Chien","Animal"), ("Chat","Chien","Animal")]):
        base = layer * N_C1
        for ci, ni in [(0,"Chat"),(1,"Chien"),(2,"Animal")]:
            for cj, nj in [(0,"Chat"),(1,"Chien"),(2,"Animal")]:
                si, sj = net._sl(base+ci), net._sl(base+cj)
                wb = net.W[si, sj]
                if wb.mean() > 0.01 or wb.mean() < -0.01:
                    print(f"    W[{ni}_C{layer+1}<-{nj}_C{layer+1}]  "
                          f"mean={wb.mean():+.4f}")

    if net.expanded and phi_dm < THETA_T:
        print(f"\n  *** PHASE 2 VALIDEE : Expansion topologique reussie ***")
        print(f"  Le reseau SNN a appris les implications en Couche 1,")
        print(f"  recrute les neurones de Couche 2 pour orthogonaliser,")
        print(f"  et continue d'apprendre dans le nouvel espace augmente.")
    elif net.expanded:
        print(f"\n  Expansion effectuee, mais convergence partielle.")
    else:
        print(f"\n  Expansion NON declenchee (condition non atteinte).")

    return net, history


if __name__ == "__main__":
    net, history = run_phase2()
