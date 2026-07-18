"""
TSO Phase 0 (V2) — Consolidation Gâtée par Φ.

Empêche l'effondrement de la représentation (Chat→Animal→Chien)
en inhibant la R-STDP locale si la co-activation implique une
tension sémantique latente (cos < 0 entre cibles).
"""
import math, random
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

ETA = 0.1
W_MAX = 1.5
TAU_E = 20.0
INHIB_FACTOR = 0.05

Z_CHAT = np.array([1.0, 0.0])
Z_CHIEN = np.array([-1.0, 0.0])
Z_ANIMAL = np.array([0.0, 1.0])

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

class LIFCluster:
    def __init__(self, name, z_target):
        self.name = name
        self.z_target = z_target
        self.rate = 0.0
        self.eligibility = 0.0
    def activate(self, intensity=1.0):
        self.rate = max(0.0, min(1.0, intensity))

class TSOCoreGated:
    def __init__(self):
        self.chat = LIFCluster("Chat", Z_CHAT)
        self.chien = LIFCluster("Chien", Z_CHIEN)
        self.animal = LIFCluster("Animal", Z_ANIMAL)
        self.W = {
            ("Chat", "Animal"): 0.1,
            ("Animal", "Chat"): 0.1,
            ("Chien", "Animal"): 0.1,
            ("Animal", "Chien"): 0.1,
            ("Chat", "Chien"): 0.01,
            ("Chien", "Chat"): 0.01,
        }

    def get_cluster(self, name):
        return {"Chat": self.chat, "Chien": self.chien, "Animal": self.animal}[name]

    def step_consolidation(self, active_labels):
        for name in active_labels:
            self.get_cluster(name).activate(1.0)

        for name, c in {"Chat": self.chat, "Chien": self.chien, "Animal": self.animal}.items():
            if c.rate > 0:
                for (pre, post), w in self.W.items():
                    if pre == name and w > 0.1:
                        post_c = self.get_cluster(post)
                        post_c.rate = max(post_c.rate, min(1.0, c.rate * w * 0.5))

        for (pre, post), w in list(self.W.items()):
            pre_c = self.get_cluster(pre)
            post_c = self.get_cluster(post)
            if pre_c.rate > 0.1 and post_c.rate > 0.1:
                semantic_sim = cos_sim(pre_c.z_target, post_c.z_target)
                pre_c.eligibility *= math.exp(-1.0 / TAU_E)
                pre_c.eligibility += pre_c.rate * post_c.rate
                if semantic_sim < 0:
                    self.W[(pre, post)] -= INHIB_FACTOR * pre_c.eligibility
                else:
                    self.W[(pre, post)] += ETA * pre_c.eligibility
                self.W[(pre, post)] = max(0.0, min(W_MAX, self.W[(pre, post)]))

    def reset_rates(self):
        self.chat.rate = 0.0
        self.chien.rate = 0.0
        self.animal.rate = 0.0
        self.chat.eligibility = 0.0
        self.chien.eligibility = 0.0
        self.animal.eligibility = 0.0

def run_gated():
    print("=" * 60)
    print("  Phase 0 (V2) — Consolidation Gâtée par Φ")
    print("  Objectif : bloquer la cascade Chat→Animal→Chien")
    print("=" * 60)

    net = TSOCoreGated()
    print(f"\n  Init:  W(Chat,Chien)={net.W[('Chat','Chien')]:.4f}")

    for epoch in range(15):
        for inp in [("Chat", "Animal"), ("Chien", "Animal")]:
            net.reset_rates()
            net.step_consolidation(inp)

        w_ca = net.W[("Chat", "Animal")]
        w_da = net.W[("Chien", "Animal")]
        w_cd = net.W[("Chat", "Chien")]
        status = "OK" if w_cd < 0.2 else "COLLAPSE"
        print(f"  Epoch {epoch:2d} | W(C→A)={w_ca:.3f} | W(D→A)={w_da:.3f} | W(C→D)={w_cd:.3f} [{status}]")

    final = net.W[("Chat", "Chien")]
    print(f"\n  >>> Résultat : W(Chat,Chien) final = {final:.4f}")
    if final < 0.2:
        print("  ✓ Gate active : les exclusifs restent séparés malgré la cascade indirecte.")
    else:
        print("  ✗ Effondrement — le Gate n'a pas suffi.")

    return final

def run_ungated():
    print("\n" + "=" * 60)
    print("  Contrôle : Même test SANS Gate (R-STDP non contrainte)")
    print("=" * 60)

    net = TSOCoreGated()
    net.W[("Chat", "Animal")] = 0.1
    net.W[("Chien", "Animal")] = 0.1
    net.W[("Chat", "Chien")] = 0.01

    print(f"\n  Init:  W(Chat,Chien)={net.W[('Chat','Chien')]:.4f}")

    for epoch in range(15):
        for inp in [("Chat", "Animal"), ("Chien", "Animal")]:
            net.reset_rates()
            # step without gate — apply LTP unconditionally
            for name in inp:
                net.get_cluster(name).activate(1.0)
            for name, c in {"Chat": net.chat, "Chien": net.chien, "Animal": net.animal}.items():
                if c.rate > 0:
                    for (pre, post), w in net.W.items():
                        if pre == name and w > 0.1:
                            net.get_cluster(post).rate = max(
                                net.get_cluster(post).rate,
                                min(1.0, c.rate * w * 0.5)
                            )
            for (pre, post), w in list(net.W.items()):
                pre_c = net.get_cluster(pre)
                post_c = net.get_cluster(post)
                if pre_c.rate > 0.1 and post_c.rate > 0.1:
                    pre_c.eligibility *= math.exp(-1.0 / TAU_E)
                    pre_c.eligibility += pre_c.rate * post_c.rate
                    net.W[(pre, post)] += ETA * pre_c.eligibility
                    net.W[(pre, post)] = max(0.0, min(W_MAX, net.W[(pre, post)]))

        w_cd = net.W[("Chat", "Chien")]
        status = "OK" if w_cd < 0.2 else "COLLAPSE"
        print(f"  Epoch {epoch:2d} | W(C→D)={w_cd:.3f} [{status}]")

    final = net.W[("Chat", "Chien")]
    print(f"\n  >>> Résultat : W(Chat,Chien) final = {final:.4f}")
    if final >= 0.2:
        print("  ✓ Effondrement confirmé : la cascade récurrente fusionne les exclusifs.")
    else:
        print("  ? Pas d'effondrement (peut-être saturation précoce).")
    return final

if __name__ == "__main__":
    f_gated = run_gated()
    f_ungated = run_ungated()
    print("\n" + "=" * 60)
    print("  BILAN :")
    if f_gated < 0.2 and f_ungated >= 0.2:
        print("  La Consolidation Gâtée par Φ résout l'effondrement.")
        print("  Le Gate est indispensable à la stabilité du réseau.")
    elif f_gated < 0.2:
        print("  Gate OK, mais le contrôle n'a pas effondré — vérifier paramètres.")
    else:
        print("  Le Gate n'a pas suffi. Augmenter INHIB_FACTOR ou revisiter.")
    print("=" * 60)
