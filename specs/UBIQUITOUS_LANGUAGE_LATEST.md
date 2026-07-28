# Ubiquitous Language — TSO (Tension-Solving Organism)

## L'organisme et son cycle

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **TSO** | Organisme artificiel bio-inspiré : un agent cognitif complet avec pulsions homéostatiques, mémoire, graphe sémantique et apprentissage moteur | Agent, bot, IA |
| **Heartbeat** | Cycle cognitif en 4 étapes (Perception → Catégorisation → Évaluation → Action) exécuté à chaque pas de temps | Tick, step, itération |
| **Bien-être** | Signal composite maximisé par l'agent : récompense filtrée + consummatory + curiosité + shaping − ΔΦ − tension chronique − pénalité déficit − parcimonie − coût métabolique | Fitness, reward total, bonheur |
| **Sommeil** | Phase hors-ligne de consolidation : rejeu bruité, résolution profonde du graphe (80 iters), élagage prototypes/arêtes/concepts inactifs | Consolidation, offline, nap |
| **Pression de sommeil** | Variable homéostatique qui s'accumule à chaque épisode ; déclenche le sommeil à 1.0 ou après 5 épisodes | Sleep debt, fatigue |

## Pulsions et homéostasie

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Hypothalamus** | Régulateur des variables homéostatiques (énergie, hydratation, température) avec dérive temporelle et modulation de la récompense | Drive system, besoin |
| **Déficit** | Écart entre l'état courant d'une variable homéostatique et son maximum (1.0) ; amplifie la valeur perçue des récompenses | Besoin, manque, privation |
| **Récompense filtrée** | Récompense externe amplifiée par le déficit : `R × (1.0 + déficit × 2.0)` | Gated reward, reward modulée |
| **Valeur consummatoire** | Plaisir immédiat de réduire un déficit : `déficit_total × 10.0` quand récompense > 0 | Consummatory reward, plaisir |
| **Coût métabolique** | Énergie dépensée par l'activité cognitive (cervelet, graphe) et motrice ; drainée de la variable énergie | Metabolic cost, effort cost |

## Perception et catégorisation

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Moustaches** | 4 capteurs de distance (N, S, E, O) — l'agent ne connaît jamais sa position absolue | Whiskers, rayons, senseurs |
| **AttractorField** | Classifieur à prototypes avec apprentissage hebbien compétitif et seuils adaptatifs par concept | Attractor, classifieur, champ d'attracteurs |
| **Prototype** | Vecteur représentant un concept dans l'espace sensoriel ; le plus proche détermine la classe | Exemplar, centroid, template |
| **Concept** | Catégorie discrète apprise par l'AttractorField ; représentée par un ou plusieurs prototypes | Classe, catégorie, nœud |
| **Seuil de nouveauté** | Distance maximale au prototype le plus proche avant création d'un nouveau concept (seuil adaptatif par concept, clampé [0.05, 0.5]) | Novelty threshold, seuil de création |
| **Élagage conceptuel** | Suppression des concepts inactifs depuis >500 pas, avec réindexation complète | Concept pruning, nettoyage |

## Attention et diagnostic

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Attention spatiale** | Gain multiplicatif sur chaque moustache basé sur l'erreur de prédiction : amplifie les dimensions surprenantes, atténue les prévisibles | Spatial attention, gain attentionnel |
| **Température du softmax** (T) | Contrôle la sélectivité de l'attention ; T=0.5 dans TSO | Attention temperature |
| **δ-clip** | Clip de |δ| dans la mise à jour de l'acteur TD : `step_a = lr · min(|δ|, delta_clip_max)`. Résout l'instabilité du TD en ligne (cf. ADR-006). | Delta clipping, gradient clipping |
| **delta_clip_max** | Seuil du δ-clip ; 5.0 par défaut dans CognitiveConfig. 0.0 = pas de clip. | delta_clip |
| **CognitiveConfig** | Struct Rust à 6 flags binant les sous-systèmes cognitifs (attractor, graph_phi, attention, episodic_curiosity, metabolic_cost, hypothalamus) + delta_clip_max. Défaut tout-à-true, zéro régression. | Config cognitif, brain config |

## Mémoire

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Mémoire de travail** | Système DualLIF (lent α=0.95 + rapide α=0.5) qui intègre la perception sur deux échelles temporelles + mémoire associative | Working memory, WM |
| **DualLIF** | Double intégrateur à fuite neuronale : contexte lent (cortex préfrontal) vs variations rapides (cortex sensoriel) | Dual Leaky Integrate-and-Fire |
| **Mémoire associative** | Stockage de vecteurs avec rappel par similarité cosinus ; analogue à l'hippocampe | Associative memory, content-addressable memory |
| **Mémoire épisodique** | Séquences de concepts stockées avec rappel par appariement suffixe-préfixe pour la prédiction | Episodic memory, trace |
| **Tampon de contexte** | Fenêtre glissante des N derniers IDs de concepts ; sert de clé de rappel épisodique | Context buffer, sliding window |
| **Surprise** | Erreur de prédiction épisodique : le concept attendu vs le concept réel ; génère une récompense de curiosité | Surprise, prediction error, curiosité |

## Graphe sémantique et tension cognitive

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Graphe sémantique** | Graphe de contraintes entre concepts : nœuds = vecteurs unitaires, arêtes = implications/exclusions pondérées | Semantic graph, knowledge graph |
| **Φ (Phi)** | Tension cognitive = somme des violations sur toutes les arêtes du graphe ; signal d'« anxiété » intrinsèque | Tension cognitive, cognitive dissonance, conflit |
| **Arête d'implication** | Poids +1 ou +2 : deux concepts doivent être similaires (cos > γ) ; +2 pour transitions à haute récompense | Implication edge, lien d'implication |
| **Arête d'exclusion** | Poids -1 : deux concepts doivent être dissimilaires (cos < ε) | Exclusion edge, lien d'exclusion |
| **Satisfaction de contraintes** | Résolution par recuit simulé : Invert / Align / Repel pour minimiser Φ | Constraint satisfaction, résolution |
| **Invert** | Action de résolution : inverse un vecteur nœud (v → −v) | Flip, inversion |
| **Align** | Action de résolution : moyenne deux vecteurs nœuds et normalise | Merge, alignement |
| **Repel** | Action de résolution : gradient tangent projectif séparant deux nœuds | Push, répulsion |
| **Démineur** | Mode agressif : plante un « drapeau » sur l'arête la plus violée à chaque pas (la supprime), réduisant Φ instantanément | Minesweeper, flag, sweep |
| **Élagage d'arêtes** | Suppression massive des arêtes à faible contribution Φ (seuil min_phi), évite la croissance quadratique | Edge pruning, nettoyage |

## Apprentissage moteur

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Cervelet** | Réseau actor-critic (linéaire ou MLP) qui sélectionne les actions pour maximiser le bien-être | Cerebellum, policy network |
| **Critique** | Tête qui estime la valeur V(s) d'un état ; apprentissage TD asymétrique | Critic, value function |
| **Acteur** | Tête qui produit la politique (distribution de probabilité sur les actions) | Actor, policy |
| **Traces d'éligibilité** | Mémoire temporaire des synapses activées par l'action choisie ; permet l'attribution de crédit temporel | Eligibility traces, decaying trace |
| **TD(λ)** | Apprentissage par différence temporelle avec traces dégradées ; compromis entre bootstrap et Monte Carlo | Temporal-difference learning |
| **Replay buffer** | Buffer circulaire (capacité 10 000) de transitions (s, a, r, s') pour mini-batch TD offline | Experience replay, buffer |

## Environnement

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **GridWorld** | Environnement en grille avec murs infranchissables ; partiellement observable (moustaches seulement) | Maze, labyrinthe, grille |
| **Configuration** | Topologie du monde : Salle vide, Couloir droit, Zigzag (L), Aléatoire (~35% murs) | Map, maze layout, config |

## Encodeur interchangeable

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **Encoder trait** | Interface Rust à une méthode requise `encode_raw()` retournant `EncodeResult { category_id, novelty, is_new }`. Permet de basculer entre catégorisation discrète et continue sans modifier TsoEngine::step(). | Encodeur, trait d'encodage |
| **AttractorEncoder** | Implémentation discrète : encapsule l'AttractorField avec seuils adaptatifs, création de concepts, élagage. Comportement historique (défaut). | Attractor wrapper, classifieur discret |
| **VaeEncoder** | Implémentation continue : VAE (64→32→8→32→64) avec centroids latents. Mappe une perception à une distribution gaussienne (µ, logσ²), échantillonne `z`, et l'assigne au centroid le plus proche. | VAE, encodeur variationnel |
| **Déterministe** | Mode `deterministic=true` : utilise `z = µ` au lieu de `z = µ + σ·ε`. Stabilité parfaite après pré-entraînement. | Deterministic, inférence µ |
| **Freeze** | Mode `freeze=true` : les centroids ne sont pas mis à jour. Utilisé en inférence seule après pré-entraînement. | Frozen, gelé |
| **VAE pré-entraîné** | VAE entraîné hors ligne sur un dataset fixe (batch, 100 epochs, ~5s), puis gelé pour l'inférence dans TSO. Résout l'instabilité de l'entraînement en ligne. | Pre-trained VAE, offline VAE |
| **centroid** | Moyenne des latents d'une catégorie. Buffer `Vec<Vec<f64>>` dans VaeEncoder. Un nouveau centroid est créé quand `distance(z, centroid) > novelty_threshold`. | Cluster, prototype latent |

## Métriques et observabilité

| Terme | Définition | Alias à éviter |
|-------|-----------|----------------|
| **MetricsSnapshot** | Struct sérialisable (serde + JSON) capturant Φ, bien-être, énergie, hydratation, température, pression de sommeil, concepts, arêtes, épisodes, steps, cycles de sommeil. Exportable via `--metrics` ou `JSON_METRICS=1`. | Métriques, snapshot |
| **tracing** | Crate de logging structuré (niveaux DEBUG/INFO/ERROR, events avec champs typés). Remplace `eprintln!` dans le code de production. Activé via `--trace` ou `TRACE=1`. | Logging, debug |
| **DEBUG step** | Event tracing émis à chaque heartbeat quand `debug_step_dump=true`. Inclut rl_signal, reward_ext, bfs_value, use_stationary_reward. | Step debug, RL trace |
| **JSON_METRICS** | Variable d'environnement activant l'export JSON des MetricsSnapshot via `serde_json::to_string()`. | JSON export |
| **well_being_weights** | Tableau `[f64; 9]` dans TsoEngine pondérant indépendamment chaque terme du bien-être. Défaut : `[1.0; 9]` (comportement historique). | Poids du bien-être, WB weights |
| **Matrice d'ablations** | Binaire `ablation_matrix.rs` : 9 termes × 5 régimes homéostatiques × 5 seeds. Mesure l'impact de l'ablation de chaque terme (poids=0). Sortie CSV. | Ablation matrix, sensitivity matrix |
| **Environment trait** | Interface `reset()`, `step(action)`, `action_space()`, `observation_dim()`. Unifie GridWorld, Minigrid, Habitat. Retourne `Array1<f64>` (pas d'allocation heap). Intégré via `Box<dyn Environment>` dans TsoEngine. | Environnement, env trait |
| **GridEnv** | Implémentation GridWorld 5×5 du trait Environment (buffer `obs_buf` réutilisé, 1.2 µs/step). | Grid environment, env 5×5 |
| **MinigridEnv** | Wrapper PyO3 vers la bibliothèque Python Minigrid. step/reset via FFI. Observation flatten (image RGB → Vec<f64> → Array1). ~50 µs/step. | MiniGrid bridge, Python env |
| **Scaling dimensionnel** | Analyse de la latence du trait Environment de dim=4 à dim=4096. Résultat : latence quasi constante (0.7–1.2 µs/step). Le goulot est l'encodeur, pas l'interface. | Dimension scaling, env scaling |

## Relations

- Un **Heartbeat** exécute 4 étapes : Perception → Catégorisation → Évaluation (Φ) → Action
- L'**Hypothalamus** fait dériver les variables à chaque heartbeat ; le **Cervelet** sélectionne l'action motrice
- L'**AttractorField** produit des **concepts** qui sont stockés dans la **Mémoire épisodique** et le **Graphe sémantique**
- Le **Graphe sémantique** accumule des **arêtes** entre concepts ; la violation de ces arêtes produit **Φ**
- Quand **Φ** dépasse `phi_threshold`, l'agent est en état d'« anxiété » et le **bien-être** diminue
- La **Mémoire de travail** (DualLIF) intègre la perception ; la **Mémoire associative** stocke des copies exactes
- La **Mémoire épisodique** prédit le prochain concept → l'erreur génère une **surprise** → récompense de **curiosité**
- L'**Attention spatiale** amplifie les moustaches où l'erreur de prédiction est forte
- Le **Sommeil** consolide : rejeu → résolution profonde → élagage prototypes/arêtes/concepts
- Le **Démineur** est un mode extrême qui supprime les arêtes violées une par une jusqu'à Φ = 0

## Exemple de dialogue

> **Dev :** Quand le Cervelet sélectionne une action motrice, comment l'Hypothalamus influence-t-il la décision ?
>
> **Expert :** L'Hypothalamus amplifie la récompense externe via le déficit — si l'énergie est basse, une récompense de nourriture vaut plus. Il ajoute aussi la valeur consummatoire (le plaisir de recharger) et la pénalité de déficit (la douleur d'être affamé). Le tout est sommé dans le bien-être.
>
> **Dev :** Et Φ là-dedans ?
>
> **Expert :** Φ est une pénalité en deux parties : ΔΦ (anxiété aiguë, transitoire) si le conflit du graphe vient d'augmenter, et −Φ²×0.005 (tension chronique, permanente). À Φ élevé, l'agent ne pense qu'à résoudre ses contradictions internes.
>
> **Dev :** Donc si le Graphe sémantique a trop d'arêtes contradictoires, l'agent explore moins et résout plus ?
>
> **Expert :** Exactement. La résolution par recuit simulé (Invert, Align, Repel) fait baisser Φ. Si ça ne suffit pas, le Démineur supprime carrément les arêtes les plus violées. Et pendant le Sommeil, on fait 80 itérations de résolution profonde + élagage.
>
> **Dev :** Et l'Attention spatiale, elle intervient où dans ce cycle ?
>
> **Expert :** Juste avant la catégorisation. La Mémoire épisodique prédit le concept attendu, et on compare le prototype prédit à la perception brute. Les moustaches où l'écart est grand sont amplifiées (gain >1.0), les autres atténuées. L'organisme « tourne la tête » vers l'inattendu.

## Ambiguïtés signalées

| Terme | Problème | Recommandation |
|-------|----------|----------------|
| **Concept** | Utilisé pour : (a) une catégorie dans l'AttractorField, (b) un nœud dans le Graphe sémantique. Ce sont les mêmes objets avec des rôles différents. | Conserver **Concept** pour l'entité unique ; clarifier le rôle par le contexte (AttractorField vs Graphe). |
| **Noeud** / **Node** | Dans `core.rs`, un nœud du graphe est un vecteur unitaire. Dans l'AttractorField, un prototype est aussi un vecteur. Ce sont les mêmes vecteurs physiques, mais les termes sont différents selon le module. | Standardiser sur **Concept** comme terme unique ; le prototype est l'embedding, le nœud est son rôle dans le graphe. |
| **Surprise** / **Curiosité** | La surprise est l'erreur de prédiction épisodique ; la curiosité est la récompense générée par cette surprise. Deux concepts distincts mais souvent utilisés de manière interchangeable. | **Surprise** = l'erreur mesurée ; **Curiosité** = récompense intrinsèque dérivée de la surprise. |
| **Sleep** / **Sommeil** | La version anglaise du code (`sleep_drive`, `sleep_pressure`) et la version française du papier. Cohérent mais à noter. | Accepter les deux ; le code est en anglais, le papier en français. |
