# Spec v2: Protocole but tournant agressif

## Problème
Switch tous les 50 épisodes → trop lent. Aliasing 2 paires → trop simple.
Résultat : tous les agents convergent en 0-2 épisodes après switch.

## Nouveau protocole

### Paramètres
- Switch tous les **10** épisodes (au lieu de 50)
- **8 buts** (4 paires d'aliasing) au lieu de 4 (2 paires)
- max_steps = **10** (au lieu de 20)
- 200 épisodes totaux → **20 phases** de 10 épisodes

### Environnement (local, pas de modif de rotating_t.rs)
- Même grille 5×5
- Observation 4D (whiskers) — inchangée
- Buts : les 8 positions extrêmes {(4,0), (0,4), (4,4), (0,0), (2,4), (4,2), (0,2), (2,0)}
- 4 paires d'aliasing : (4,0)=(0,4) ?, (4,4)=(0,0) ?, (2,4)=(4,2) ?, (0,2)=(2,0) ?
- Les 8 buts défilent en séquence fixe, pas de répétition avant les 8

### Métriques
- **Reward par phase de 10** : moyenne des 10 épisodes
- **Pente intra-phase** : reward(ép5-9) − reward(ép0-4) → vitesse d'adaptation
- **Reward plateau** : reward moyen des 5 dernières phases (steady-state)
- **Reward drop** : chute au switch = reward(ép1 après switch) − reward(ép10 avant switch)
- **Forgetting** : quand un même but revient après 8 phases, le reward est-il meilleur ?

### Hypothèses
1. Avec switch tous les 10, les agents sans mémoire (linear) ne peuvent pas suivre
2. La mémoire épisodique (TSO + épisodique) doit permettre un transfert entre phases
3. L'AttractorField seul ne suffit pas si les buts sont trop nombreux
4. Le Φ gating (si activé) devrait fluctuer avec les switchs
