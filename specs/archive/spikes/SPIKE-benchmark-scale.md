# Spike: Benchmarks plus grands que MiniGrid 7×7

## Question
TSO montre un avantage sur 147D. L'avantage tient-il sur des dimensions
encore plus grandes ?

## Options explorées

### 1. MiniGrid paramétrique (Rust pur, faisable)
- Actuel : 7×7 = 147D RGB (3×7×7)
- Proposition : W/H paramétrables → 13×13 = 507D, 19×19 = 1083D
- Avantage : Rust pur, zéro dépendance, réutilise l'Environment trait
- Coût : 30 min de refactor

### 2. MiniWorld (Rust, 3D, OpenGL)
- librairie Rust miniworld-rs : immature, peu maintenue
- Bundler GL → dépendance système
- **YAGNI** — trop d'overhead pour un benchmark

### 3. Procgen (Python)
- 16 environnements procéduraux
- Nécessite Python + PyTorch → incompatible avec le kernel Rust pur
- **YAGNI** — sauf si bindings PyO3 (coût élevé)

### 4. Griddly (Rust/C++)
- Gym-compatible, Rust bindings existants
- Supporte grilles arbitraires, sprites, SVG
- Prometteur mais nécessite compilation C++
- **À garder en tête** pour phase 2 si validation nécessaire

### 5. Atari (ALE)
- Rust ALE bindings (ale-rs)
- 84×84×4 = 28 224D → trop pour ndarray sans CNN
- **YAGNI**

## Recommandation
**MiniGrid paramétrique** — le plus simple, le plus rapide, le plus pertinent.
Ajouter W/H configurable à MiniGridEnv, puis mesurer TSO vs linear sur
13×13 et 19×19. Si l'avantage TSO grandit avec la dimension, la thèse
est consolidée. Si l'avantage plafonne, on sait que TSO a un plafond.

## Plan
1. Refactor MiniGridEnv : W/H en const → paramètres constructeur
2. Garder DoorKey 7×7 comme défaut (backward compat)
3. Ajouter `bin/bench_scale.rs` qui run 7×7, 13×13, 19×19
4. Comparer TSO vs linear à chaque taille
