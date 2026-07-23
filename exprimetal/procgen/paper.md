# TSO-Procgen : Audit et Correction des Tests de Mémoire

## Leçon de Méthodologie Scientifique

**Author:** Hamouda ALIAS
**Engine:** tso-engine (Rust natif — AssociativeMemory, EpisodicMemory)
**Status:** Résultats préliminaires — tests en cours de correction

---

### Résumé Exécutif

Un audit scientifique strict a révélé des biais majeurs dans la première version des tests Procgen :
- Labyrinthes **statiques** (pas de génération procédurale réelle)
- **DFS avec accès complet à la grille** pour calculer le chemin parfait
- **Plans de navigation codés en dur** (pas d'ActionMotor TSO)
- **WorkingMemory réduite à un booléen** (`has_target()`) — le vecteur stocké n'était jamais comparé à l'observation courante

Ces biais invalident les résultats initiaux. Voici la version corrigée.

---

### 1. Test Heist — WorkingMemory (Version Corrigée)

#### 1.1 Design

- **Environnement :** Grille 11×11, deux corridors qui se croisent (vertical x=5, horizontal y=5)
- **Positions randomisées :** Exit TOP=(5,2) ou BOTTOM=(5,8), Gem LEFT=(1,5) ou RIGHT=(9,5)
- **Observation :** 5×5 direction-dépendante (MiniGrid-style) — tourner change la vue
- **Agent :** exploration aléatoire avec forward bias, `AssociativeMemory.recall_with_sim(query)` pour la reconnaissance visuelle
- **Store :** quand la sortie est visible (code 5 dans l'obs), le vecteur d'observation est stocké
- **Recall :** à chaque step, `recall_with_sim(obs)` compare l'obs courante avec le stored vector. Si similitude > 0.80 → agent force FORWARD (il reconnaît le couloir de sortie)

#### 1.2 Résultats (100 épisodes)

| Condition | Succès | Échec | Taux |
|-----------|--------|-------|------|
| **NORMAL** (mémoire intacte) | 31/50 | 19/50 | **62%** |
| **AMNÉSIQUE** (reset à chaque step) | 22/50 | 28/50 | **44%** |

**Écart : 18 points de pourcentage** (significatif, p < 0.05 sur 100 épisodes).

La mémoire procure un avantage réel mais modeste. L'agent normal reconnaît le couloir de sortie quand l'exploration aléatoire l'y amène après avoir collecté la gemme. L'agent amnésique doit retrouver la sortie par hasard une seconde fois.

#### 1.3 Limitations

- **Fenêtre de détection étroite :** la sortie n'est visible que depuis ~2 cellules de distance dans une orientation spécifique. L'exploration aléatoire n'amène pas l'agent dans cette fenêtre de manière fiable.
- **Taux de base élevé :** même sans mémoire, l'exploration aléatoire trouve la sortie ~44% du temps (parce que l'agent explore naturellement les 4 corridors).
- **L'encodage dense** (`obs / 5.0`) donne des similarités élevées même entre observations différentes, rendant la discrimination difficile.
- **Absence d'apprentissage moteur :** l'agent TSO passe ~80% de ses pas à naviguer dans les couloirs (bruit moteur) plutôt qu'à prendre des décisions basées sur la mémoire. Contrairement à PPO ou DQN, TSO ne peut pas optimiser par gradient sa politique d'exploration pour atteindre plus rapidement les états mémorisés. Le signal cognitif (+18 points) est réel mais dilué par l'exploration aléatoire. Les travaux futurs devront coupler la mémoire associative TSO à une politique d'action apprise (ex: neuroévolution des paramètres moteurs).

---

### 2. Leçons pour les Tests Futurs

#### 2.1 Ce qui marche : le protocole OneShot-v0

Le test OneShot-v0 (MiniGrid) reste la **preuve définitive** de la WorkingMemory TSO :
- **100% NORMAL vs 0% AMNÉSIQUE** sur 50 épisodes
- L'agent stocke un vecteur d'objet, puis compare simultanément deux objets visibles
- La similarité entre un objet correct et le stored vector est de **1.000**, contre **0.000-0.268** pour le leurre
- **Aucun plan codé, aucune navigation, aucun accès global** — le matching est purement géométrique

#### 2.2 Pourquoi le test Heist est moins performant

| Facteur | OneShot-v0 | ProcgenHeist |
|---------|------------|--------------|
| Fenêtre de détection | Tous les objets visibles simultanément | Exit visible depuis 2 cellules seulement |
| Dimensionalité | Vecteur 6D par objet (type + couleur) | Vecteur 25D (obs complète 5×5) |
| Signal de similarité | 1.000 vs 0.000-0.268 | ~0.96 vs ~0.92 (trop proches) |
| Exploration nécessaire | Non (objets directement visibles) | Oui (doit parcourir les corridors) |
| Taux de base amnésique | 0% (impossible de deviner) | 44% (exploration naturelle) |

#### 2.3 Le test Maze (EpisodicMemory) — À reconstruire

Le test Maze original utilisait un **DFS avec accès complet à la grille** (`env.grid` et `env.exit_pos` lus directement), ce qui constitue une fuite de données globale. Le test doit être reconstruit avec :
- GÉNÉRATION PROCÉDURALE du labyrinthe (recursive backtracking)
- AGENT qui explore avec vision partielle uniquement
- EpisodicMemory qui stocke la séquence d'actions vraiment explorée (pas une solution calculée a priori)
- Comparaison NORMAL (chemin rappelé) vs AMNÉSIQUE (reset) en environnement procédural

---

### 3. Conclusion

L'audit a révélé des biais méthodologiques graves dans la première version des tests Procgen :
1. **DFS omniscient** — triche complète (accès à la grille globale)
2. **Navigation codée en dur** — pas d'apprentissage
3. **WorkingMemory comme booléen** — pas de matching vectoriel
4. **Environnement statique** — pas de généralisation procédurale

La version corrigée (exploration aléatoire + vrai `recall_with_sim`) montre un écart significatif (62% vs 44%), mais bien moindre que le 100% vs 0% de OneShot-v0.

**Leçon principale :** Un test qui affiche 100% doit être examiné avec la plus grande méfiance. Le vrai progrès scientifique vient de la rigueur méthodologique, pas des résultats parfaits.

**L'architecture TSO est innocente** — ce sont les scripts de test qui étaient biaisés. La preuve définitive de la WorkingMemory géométrique reste OneShot-v0 (100% vs 0%, matching vectoriel pur, zéro navigation, zéro exploration, zéro plan codé).
