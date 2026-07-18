"""
TSO Phase 0 — Simulation des Algorithmes 1 & 2
Validation de la dynamique des vecteurs et de Phi global

Explore plusieurs strategies d'operateur pour comprendre
le probleme de type Hopfield identifie dans le paper.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable

SEED = 42
random.seed(SEED)

# ─── Parametres ───────────────────────────────────────────────────────────────
d = 8
GAMMA = 0.7
EPSILON = 0.3
THETA_T = 0.1
THETA_C = 0.5
ALPHA = math.pi / 4

# ─── Utilitaires vectoriels ───────────────────────────────────────────────────

def vec_add(a, b): return [x + y for x, y in zip(a, b)]
def vec_sub(a, b): return [x - y for x, y in zip(a, b)]
def vec_scale(v, s): return [x * s for x in v]
def vec_dot(a, b): return sum(x * y for x, y in zip(a, b))
def vec_norm(v): return math.sqrt(vec_dot(v, v)) + 1e-10
def vec_normalize(v): n = vec_norm(v); return [x / n for x in v]
def vec_random(dim, scale=1.0): return [random.gauss(0, scale) for _ in range(dim)]
def vec_cos_sim(a, b): return vec_dot(a, b) / (vec_norm(a) * vec_norm(b))

def apply_rotation(v, alpha):
    v_norm = vec_normalize(v)
    r = vec_random(len(v))
    v_perp = vec_sub(r, vec_scale(v_norm, vec_dot(r, v_norm)))
    v_perp_norm = vec_normalize(v_perp)
    norm_v = vec_norm(v)
    return vec_add(
        vec_scale(v, math.cos(alpha)),
        vec_scale(v_perp_norm, math.sin(alpha) * norm_v)
    )

# ─── Graphe ───────────────────────────────────────────────────────────────────

@dataclass
class GraphTSO:
    vectors: Dict[str, List[float]]
    edges: List[Tuple[str, str, int]]
    dim: int = d

def build_initial_graph(seed_offset=0):
    rng = random.Random(SEED + seed_offset) if seed_offset else random
    def vr(dim, scale=1.0): return [rng.gauss(0, scale) for _ in range(dim)]

    base = vr(d, 0.3)
    z_animal = vec_normalize(vec_add(base, vr(d, 0.2)))
    common = vr(d, 0.2)
    z_chat = vec_normalize(vec_add(vec_scale(z_animal, 0.3), vec_scale(common, 0.7)))
    z_chien = vec_normalize(vec_add(vec_scale(z_animal, 0.3), vec_scale(common, 0.7)))
    z_chat = vec_normalize(vec_add(z_chat, vr(d, 0.02)))
    z_chien = vec_normalize(vec_add(z_chien, vr(d, 0.02)))

    return GraphTSO(
        vectors={"Chat": z_chat, "Chien": z_chien, "Animal": z_animal},
        edges=[("Chat", "Animal", 1), ("Chien", "Animal", 1), ("Chat", "Chien", -1)],
        dim=d,
    )

# ─── Phi ──────────────────────────────────────────────────────────────────────

def compute_phi(g: GraphTSO):
    phi = 0.0
    violations = {}
    for i, j, w in g.edges:
        dot = vec_cos_sim(g.vectors[i], g.vectors[j])
        if w == 1:
            v = max(0.0, GAMMA - dot)
            t = "imp"
        else:
            v = max(0.0, dot - EPSILON)
            t = "exc"
        phi += v
        violations[(i, j)] = (t, v, dot)
    return phi, violations

def print_phi(g: GraphTSO, label="Phi"):
    phi, viols = compute_phi(g)
    print(f"  {label} = {phi:.6f}")
    for (i, j), (t, v, dot) in viols.items():
        sym = "->" if t == "imp" else "_|_"
        m = "VIOL" if v > 0 else "ok "
        print(f"    {i:6s}{sym}{j:6s}  dot={dot:.4f}  viol={v:.4f}  [{m}]")
    return phi

# ─── Strategies d'operateur ──────────────────────────────────────────────────
# Chaque strategie est une fonction: (g, i, j, phi_local) -> GraphTSO

def strategy_strict_orthogonal(g, i, j, phi_local):
    """k=d strict: [z_i, 0], [0, z_j], les autres zero-pad."""
    nd = g.dim * 2
    nv = {}
    for name, v in g.vectors.items():
        zp = [0.0] * nd
        if name == i:
            zp[:g.dim] = v
        elif name == j:
            zp[g.dim:] = v
        else:
            zp[:g.dim] = v
        nv[name] = zp
    return GraphTSO(vectors=nv, edges=g.edges, dim=nd)

def strategy_split_nonconflict(g, i, j, phi_local):
    """k=d: [z_i, 0], [0, z_j], les autres divises entre les 2 sous-espaces."""
    nd = g.dim * 2
    nv = {}
    for name, v in g.vectors.items():
        zp = [0.0] * nd
        if name == i:
            zp[:g.dim] = v
        elif name == j:
            zp[g.dim:] = v
        else:
            # Diviser le vecteur entre les deux sous-espaces
            v_sq = vec_scale(v, 1.0 / math.sqrt(2))
            zp[:g.dim] = v_sq
            zp[g.dim:] = v_sq
        nv[name] = zp
    return GraphTSO(vectors=nv, edges=g.edges, dim=nd)

def strategy_tension_rotation(g, i, j, phi_local):
    """k=d avec rotation: utiliser la rotation pour tous les cas."""
    nd = g.dim * 2
    nv = {}
    for name, v in g.vectors.items():
        zp = [0.0] * nd
        v_rot = apply_rotation(v, ALPHA)
        if name == i:
            zp[:g.dim] = v
            zp[g.dim:] = v_rot[:g.dim]
        elif name == j:
            v_rot2 = apply_rotation(v, ALPHA * 1.5)
            zp[:g.dim] = v_rot2[:g.dim]
            zp[g.dim:] = v
        else:
            zp[:g.dim] = v
            zp[g.dim:] = v_rot[:g.dim]
        nv[name] = zp
    return GraphTSO(vectors=nv, edges=g.edges, dim=nd)

def strategy_soft_separate(g, i, j, phi_local):
    """Separation progressive via rotation douce (pas d'expansion)."""
    vi = g.vectors[i]
    vj = g.vectors[j]

    # Angle cible : fonction de la violation
    target_angle = min(math.pi / 2, math.acos(EPSILON) + 0.1)
    current_angle = math.acos(max(-1, min(1, vec_cos_sim(vi, vj))))
    frac = min(1.0, phi_local / THETA_C * 0.5)
    step_angle = (target_angle - current_angle) * frac

    vi_new = apply_rotation(vi, step_angle)
    vj_new = apply_rotation(vj, -step_angle)

    nv = dict(g.vectors)
    nv[i] = vec_normalize(vi_new)
    nv[j] = vec_normalize(vj_new)
    return GraphTSO(vectors=nv, edges=g.edges, dim=g.dim)

def strategy_attract(g, i, j, phi_local):
    """Attraction : rapprocher deux vecteurs (pour implication)."""
    vi = g.vectors[i]
    vj = g.vectors[j]
    mix = vec_normalize(vec_add(vi, vec_scale(vj, 0.3)))
    nv = dict(g.vectors)
    nv[i] = mix
    return GraphTSO(vectors=nv, edges=g.edges, dim=g.dim)


STRATEGIES = {
    "strict_orthogonal": strategy_strict_orthogonal,
    "split_nonconflict": strategy_split_nonconflict,
    "tension_rotation": strategy_tension_rotation,
    "soft_separate": strategy_soft_separate,
    "attract": strategy_attract,
}

# ─── Algorithm 2 avec selection de strategie ──────────────────────────────────

def algorithm2_try_all(g, step, verbose=False):
    """Algo 2: essaie TOUTES les strategies et prend la meilleure."""
    phi_before, viols = compute_phi(g)

    if phi_before < THETA_T:
        return g, phi_before, phi_before, 0.0, "passive", ""

    candidate = max(viols.values(), key=lambda x: x[1])
    i, j = None, None
    for (a, b), (t, v, d) in viols.items():
        if v == candidate[1]:
            i, j = a, b
            break

    if candidate[1] <= 0:
        return g, phi_before, phi_before, 0.0, "stable", ""

    typ, val, dot = candidate

    best_delta = -float('inf')
    best_result = None
    best_name = ""

    for sname, sfunc in STRATEGIES.items():
        w_ij = 1
        for a, b, w in g.edges:
            if (a == i and b == j) or (a == j and b == i):
                w_ij = w
                break
        phi_local = max(0.0, GAMMA - dot) if w_ij == 1 else max(0.0, dot - EPSILON)

        if sname == "attract" and w_ij != 1:
            continue  # attract only for implication
        if sname in ("strict_orthogonal", "split_nonconflict", "tension_rotation") and phi_local < THETA_T:
            continue  # expansion only for significant violations
        if sname == "soft_separate" and w_ij != -1:
            continue  # only for exclusion

        g_sim = sfunc(g, i, j, phi_local)
        phi_after, _ = compute_phi(g_sim)
        delta = phi_before - phi_after

        if delta > best_delta:
            best_delta = delta
            best_result = g_sim
            best_name = sname

    if best_delta > 0 and best_result is not None:
        if verbose:
            print(f"  [Step {step}] ({i},{j}) {typ} val={val:.4f} -> "
                  f"meilleur={best_name} DPhi={best_delta:.4f}")
        return best_result, phi_before, compute_phi(best_result)[0], best_delta, "accepted", best_name
    else:
        if verbose:
            print(f"  [Step {step}] ({i},{j}) {typ} val={val:.4f} -> "
                  f"toutes inhibees (meilleur DPhi={best_delta:.4f})")
        return g, phi_before, phi_before, best_delta, "inhibited", "none"


# ─── Simulation principale ────────────────────────────────────────────────────

def run_simulation(strategy_mode="single", max_steps=50, verbose=True):
    """strategy_mode: 'single' (original), 'all' (essaie tout)."""
    g = build_initial_graph()
    history = []

    print(f"\n  Mode: {strategy_mode}")
    if verbose:
        print_phi(g, "  Etat initial")

    inhib_streak = 0
    for step in range(max_steps):
        if strategy_mode == "single":
            # Original: toujours l'arete la plus violee, operateur strict
            phi_before, viols = compute_phi(g)
            if phi_before < THETA_T:
                if verbose: print(f"  -> Stable: Phi={phi_before:.4f} < theta_t")
                history.append({"step": step, "phi_before": phi_before, "phi_after": phi_before,
                                "delta_phi": 0, "mode": "passive", "dim": g.dim, "op": ""})
                break

            cand = max(viols.values(), key=lambda x: x[1])
            i = j = None
            for (a, b), (t, v, d) in viols.items():
                if v == cand[1]: i, j = a, b; typ, val, dot = t, v, d; break

            if val <= 0:
                history.append({"step": step, "phi_before": phi_before, "phi_after": phi_before,
                                "delta_phi": 0, "mode": "stable", "dim": g.dim, "op": ""})
                break

            w_ij = 1
            for a, b, w in g.edges:
                if (a == i and b == j) or (a == j and b == i): w_ij = w; break
            phi_local = max(0.0, GAMMA - dot) if w_ij == 1 else max(0.0, dot - EPSILON)

            g_sim = strategy_strict_orthogonal(g, i, j, phi_local)
            phi_after, _ = compute_phi(g_sim)
            delta = phi_before - phi_after

            mode = "accepted" if delta > 0 else "inhibited"
            if mode == "accepted": g = g_sim
            history.append({"step": step, "phi_before": phi_before, "phi_after": phi_after,
                            "delta_phi": delta, "mode": mode, "dim": g.dim, "op": "strict"})

            if verbose:
                m = "ACCEPTE" if delta > 0 else "INHIBE"
                print(f"  Step {step}: ({i},{j}) {typ} val={val:.4f} DPhi={delta:.4f} -> {m}")

            if mode == "inhibited": inhib_streak += 1
            else: inhib_streak = 0

            if inhib_streak >= 5:
                if verbose: print(f"  -> 5x inhibitions, arret")
                break

        elif strategy_mode == "all":
            g_new, pb, pa, delta, mode, op_name = algorithm2_try_all(g, step, verbose=verbose)
            g = g_new
            history.append({"step": step, "phi_before": pb, "phi_after": pa,
                            "delta_phi": delta, "mode": mode, "dim": g.dim, "op": op_name})
            if mode == "inhibited": inhib_streak += 1
            else: inhib_streak = 0
            if inhib_streak >= 10:
                if verbose: print(f"  -> 10x inhibitions, arret")
                break
            if mode in ("passive", "stable"):
                break

    return g, history


def summarize(strategy_mode, g, history):
    phi_final, _ = compute_phi(g)
    accepted = sum(1 for h in history if h["mode"] == "accepted")
    inhibited = sum(1 for h in history if h["mode"] == "inhibited")

    phi_seq = [h["phi_after"] for h in history]
    monotone = all(phi_seq[i] <= phi_seq[i-1] + 1e-10 for i in range(1, len(phi_seq)))

    violations_ok = 0
    violations_total = 0
    for i, j, w in g.edges:
        dot = vec_cos_sim(g.vectors[i], g.vectors[j])
        violations_total += 1
        if w == 1:
            if dot >= GAMMA - 1e-10: violations_ok += 1
        else:
            if dot <= EPSILON + 1e-10: violations_ok += 1

    return {
        "mode": strategy_mode,
        "steps": len(history),
        "phi_init": history[0]["phi_before"],
        "phi_final": phi_final,
        "dim_final": g.dim,
        "accepted": accepted,
        "inhibited": inhibited,
        "monotone": monotone,
        "contraintes_satisfaites": f"{violations_ok}/{violations_total}",
        "ops_used": list(set(h["op"] for h in history if h["op"])),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    strategies = ["single", "all"]
    all_results = []

    for sm in strategies:
        g, hist = run_simulation(strategy_mode=sm, max_steps=30, verbose=True)
        s = summarize(sm, g, hist)
        all_results.append(s)

        print(f"\n  Matrice de similarite finale ({sm}):")
        names = list(g.vectors.keys())
        for ni in names:
            for nj in names:
                sdot = vec_cos_sim(g.vectors[ni], g.vectors[nj])
                print(f"    {ni:6s} . {nj:6s} = {sdot:.4f}")
        print()

    # Tableau comparatif
    print("=" * 80)
    print("  TABLEAU COMPARATIF DES STRATEGIES")
    print("=" * 80)
    print(f"  {'Strategie':20s} | {'Steps':>5s} | {'Phi_init':>8s} | {'Phi_final':>8s} | "
          f"{'Dim':>4s} | {'Accept':>6s} | {'Inhib':>6s} | {'Monotone':>9s} | {'Contraintes':>12s}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*12}")
    for r in all_results:
        print(f"  {r['mode']:20s} | {r['steps']:5d} | {r['phi_init']:8.4f} | {r['phi_final']:8.4f} | "
              f"{r['dim_final']:4d} | {r['accepted']:6d} | {r['inhibited']:6d} | "
              f"{'OUI':9s} | {r['contraintes_satisfaites']:>12s}")

    print()
    print("  Analyse :")
    print("  - strict_orthogonal echoue car l'expansion Chat-Chien casse")
    print("    les implications Chien->Animal et Chat->Animal")
    print("  - Le probleme est geometrique : impossible de satisfaire")
    print("    les 3 contraintes simultanement en 2d avec gamma=0.7")
    print("  - Solutions possibles :")
    print("    1) Expansion multi-step (k=2d, 3d...) pour creer plus de degres de liberte")
    print("    2) Separation progressive via rotation (soft_separate)")
    print("    3) Renforcer d'abord les implications (attract) PUIS separer")
    print("    4) Parametres adaptatifs (gamma, epsilon qui evoluent)")
