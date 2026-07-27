# 🧠 Guide des Skills pi pour TSO — Avec Exemples Concrets
> Projet : Tension-Solving Organism — Architecture cognitive en Rust

Chaque ligne montre : la skill → à quoi elle sert → **comment l'utiliser concrètement sur TSO**

---

## 🎯 Planification & Stratégie

```
┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill                │ Utile pour TSO                                                                               │ Exemple concret sur TSO                                                                           │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ survey-context       │ Cartographier où en est le projet, phase du cycle, prochaine étape — à faire à chaque retour │ ➜ Lancer survey-context → il lit specs/, paper.md, le code Rust → te dit si t'es en phase          │
│                      │                                                                                              │   Discover, Design, Plan ou Build, et quoi faire ensuite                                            │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ session-state        │ Suivre les décisions d'implémentation dans specs/state.yaml pour ne pas perdre le fil        │ ➜ Avant d'attaquer un nouveau module (ex: refactor de l'attention), on écrit l'état courant        │
│                      │                                                                                              │   dans specs/state.yaml pour que pi sache où t'en es à la prochaine session                       │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ define-language      │ Extraire un glossaire du domaine TSO (Φ, attracteurs, homéostasie, pulsions…)               │ ➜ DÉMONSTRÉ → a produit specs/UBIQUITOUS_LANGUAGE_LATEST.md avec 40 termes du domaine TSO,        │
│                      │                                                                                              │   4 ambiguïtés signalées (Concept ≠ Nœud ≠ Prototype, Surprise ≠ Curiosité)                       │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ deepen-architecture  │ Trouver des opportunités de refactoring dans le moteur Rust (le code est dense)              │ ➜ Scanne core.rs, attractor.rs, cerebellum.rs → propose de :                                     │
│                      │                                                                                              │   • Factoriser la résolution de contraintes (core.rs a 3 méthodes qui se chevauchent)             │
│                      │                                                                                              │   • Extraire les seuils adaptatifs de l'AttractorField en paramètres configurables                │
│                      │                                                                                              │   • Unifier la représentation des vecteurs (f32 vs f64 éparpillé dans le code)                    │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ spike-prototype      │ Explorer une idée rapidement sans engagement (nouveau module, variante d'algo)               │ ➜ Tu veux essayer un nouveau mécanisme d'attention (softmax tempéré Laplace au lieu de           │
│                      │                                                                                              │   Gaussian) ? → spike-prototype code un prototype jetable, produit des notes dans                 │
│                      │                                                                                              │   specs/archive/spikes/SPIKE-attention-laplace.md — s'il marche pas, on jette                     │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ grill-me             │ Stresser un plan architectural par des questions agressives avant de coder                    │ ➜ Tu veux ajouter un module d'émotions ? → grill-me pose 20 questions :                          │
│                      │                                                                                              │   « Comment les émotions interagissent-elles avec Φ ? » « Sont-elles apprises ou codées en dur ? »│
│                      │                                                                                              │   « Quel est le coût métabolique ? » → jusqu'à ce que le plan tienne la route                    │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ model-domain         │ Confronter un plan au modèle de domaine existant, affiner la terminologie                     │ ➜ Tu veux ajouter un module « Hormones » → model-domain le confronte au glossaire existant        │
│                      │                                                                                              │   et à l'Hypothalamus pour éviter les duplications et incohérences terminologiques                │
└──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Qualité & Tests

```
┌─────────────────────┬──────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill               │ Utile pour TSO                                                                   │ Exemple concret sur TSO                                                                           │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ setup-environment   │ S'assurer que Rust/cargo et les dépendances sont prêtes                          │ ➜ Vérifie que rustc, cargo, les bonnes versions sont installées, que le projet compile avec        │
│                     │                                                                                  │   cargo build, que les binaires (phase1b, debog_rl, weakness_game) sont buildables                 │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ develop-tdd         │ Développer en TDD (red-green-refactor) — pile pour un projet de cette complexité │ ➜ Tu ajoutes un nouveau type d'arête au graphe sémantique :                                       │
│                     │                                                                                  │   1. Écris un test qui échoue : test_implication_forte_penalite_double()                          │
│                     │                                                                                  │   2. Implémente dans core.rs                                                                      │
│                     │                                                                                  │   3. Refactor : extrais la constante de pénalité en paramètre configurable                        │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ enforce-first       │ Vérifier que les tests respectent le rubric F.I.R.S.T (Fast, Isolated, etc.)     │ ➜ Passe en revue les tests existants dans tso-engine/src/ : les tests qui mockent GridWorld        │
│                     │                                                                                  │   sont-ils isolés ? Rapides ? Répétables ? Auto-vérifiants ?                                      │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ plan-tests          │ Concevoir une architecture de test adaptée au risque avant d'implémenter          │ ➜ Avant d'ajouter la parallélisation du graphe : planifie des tests unitaires (resolve_parallel), │
│                     │                                                                                  │   d'intégration (2 threads, Φ converge) et de performance (timeout si >100ms)                     │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ run-evals           │ Définir des évaluations (capability + régression) avant de coder une feature     │ ➜ Avant de modifier le mécanisme de sommeil : définit un eval qui mesure Φ après sommeil,         │
│                     │                                                                                  │   nombre de concepts élagués, temps d'exécution → lance-le avant/après la modification            │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ kickoff-branch      │ Créer une branche propre + worktree pour chaque feature sans polluer main        │ ➜ Avant de coder le nouveau module d'attention : git worktree add ../tso-attention feat/attention  │
│                     │                                                                                  │   → tu travailles isolé, main reste propre, les tests passent toujours sur main                   │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ release-branch      │ Merge, PR, nettoyage — livrer proprement                                         │ ➜ Feature finie : release-branch fusionne proprement, option solo (land-branch.sh) ou PR via gh    │
│                     │                                                                                  │   , nettoie le worktree                                                                            │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ audit-code          │ Auto-check qualité avant review (conventions, coverage, types)                    │ ➜ Passe la checklist : le code respecte-t-il les conventions Rust ? Y a-t-il des unwrap() dangereux│
│                     │                                                                                  │   ? Des fonctions > 50 lignes ? Des clones() évitables ?                                         │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ request-review      │ Review indépendante par un second agent — second regard objectif                  │ ➜ Après avoir fini le module d'attention spatiale, un agent reviewer frais (qui n'a pas vu le     │
│                     │                                                                                  │   code avant) critique : API, nommage, edge cases oubliés                                          │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ respond-review      │ Appliquer systématiquement les retours de review                                 │ ➜ Le reviewer a trouvé 4 problèmes : 2 à corriger, 1 à discuter, 1 faux positif →                  │
│                     │                                                                                  │   respond-review les traite un par un, vérifie que les tests passent après chaque fix              │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ simulate-agents     │ Lancer un Mock User + Auditor contre une feature avant la review humaine          │ ➜ Nouveau mode « Démineur v2 » : simulate-agents lance un agent qui joue l'utilisateur             │
│                     │                                                                                  │   (lance le binaire, vérifie la sortie) + un agent auditeur (vérifie le code) avant toi           │
└─────────────────────┴──────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🐛 Debug & Investigation

```
┌─────────────────────┬──────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill               │ Utile pour TSO                                                                   │ Exemple concret sur TSO                                                                           │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ investigate-bug     │ Investiguer un bug pas à pas dans le moteur — lire le code, trouver la cause      │ ➜ Φ ne descend jamais sous 0.5 pendant le sommeil → investigate-bug trace le code :                │
│                     │                                                                                  │   regarde resolve_anneal, pruner, vérifie les paramètres, lit l'output du binaire de test          │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ diagnose-root       │ RCA en 4 phases (reproduire → isoler → hypothèse → vérifier)                     │ ➜ Le Cervelet diverge après 100 épisodes → reproduis avec le binaire debug_rl,                    │
│                     │                                                                                  │   isole le bug dans reinforce_td, formule l'hypothèse « poids qui explosent », vérifie             │
│                     │                                                                                  │   avec une version patchée                                                                         │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ validate-fix        │ Prouver qu'un fix marche avant de déclarer done (test + typecheck + lint)         │ ➜ Fix du bug de divergence du Cervelet → re-run le test qui échouait, run tout le test suite,      │
│                     │                                                                                  │   cargo clippy, cargo check → tout passe → ajoute un test de régression                           │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ quick-fix           │ Fast-path pour correction de données uniquement (sans TDD, sans branche)           │ ➜ Un seuil par défaut dans attractor.rs est mal calibré (0.3 au lieu de 0.05) →                    │
│                     │                                                                                  │   quick-fix corrige la constante, fait un cargo test, push direct                                 │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ fix-bug             │ Orchestrateur complet : investigate-bug → develop-tdd → validate-fix              │ ➜ Bug signalé « l'agent ne dort jamais » → fix-bug enchaîne les 3 skills automatiquement          │
│                     │                                                                                  │   : investigation, fix en TDD, validation                                                          │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ inspect-quality     │ Session QA interactive où tu rapportes des bugs, l'agent les enregistre           │ ➜ Tu : « Quand l'agent tourne en mode Zigzag, il se bloque au bout de 50 pas » →                   │
│                     │                                                                                  │   inspect-quality explore le code, enregistre le bug dans specs/bugs/registry.yaml                │
└─────────────────────┴──────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Production & CI

```
┌──────────────────────┬───────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill                │ Utile pour TSO                                                                    │ Exemple concret sur TSO                                                                           │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ wire-ci              │ Ajouter CI (GitHub Actions) pour le projet Rust — cargo build, test, clippy        │ ➜ Crée .github/workflows/rust.yml : cargo build, cargo test, cargo clippy sur chaque push,         │
│                      │                                                                                   │   avec matrix nightly/stable                                                                       │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ wire-observability   │ Logging structuré dans le moteur — tracer les pulsions, Φ, bien-être en temps réel │ ➜ Ajoute du logging structuré (tracing crate) : chaque heartbeat loggue Φ, bien_être,              │
│                      │                                                                                   │   déficit_hydratation, action choisie → tu visualises avec jaeger ou une sortie JSON               │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ guard-git            │ Bloquer les push/force-push/clean accidentels — hooks de sécurité                  │ ➜ Installe des hooks qui bloquent git push sur main sauf si feature branch,                       │
│                      │                                                                                   │   bloquent git reset --hard si des unstaged changes existent                                      │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ hook-commits         │ Husky + lint-staged + Prettier + tests en pré-commit                               │ ➜ Avant chaque commit : cargo fmt, cargo clippy --fix, cargo test — si un test échoue,            │
│                      │                                                                                   │   le commit est bloqué                                                                             │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ deploy               │ Pipeline build → vérifier → déployer → health-check                                │ ➜ Si TSO devient un service : build le binaire, scp sur le serveur, systemctl restart,             │
│                      │                                                                                   │   curl /health → 200 OK                                                                            │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ smoke-test           │ Health-check post-déploiement contre une URL live                                 │ ➜ Après déploiement : vérifie que le binaire répond, que les metrics endpoints sont up,            │
│                      │                                                                                   │   que le mode Démineur tourne sans crash                                                           │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ harden-vps           │ Durcir un VPS production : firewall, fail2ban, SSH, backups, monitoring            │ ➜ Si TSO tourne sur un VPS : UFW (port 22 + port TSO only), fail2ban SSH, unattended-upgrades,    │
│                      │                                                                                   │   backup automatisé du modèle tso_model.bin                                                        │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ validate-contracts   │ Vérifier la cohérence des données entre couches API/schéma/migration               │ ➜ Si TSO exporte des données : vérifie que le format du modèle sauvegardé (tso_model.bin) est      │
│                      │                                                                                   │   cohérent avec la structure en mémoire, ajoute un test de roundtrip                              │
└──────────────────────┴───────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 Documentation

```
┌─────────────────────┬──────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill               │ Utile pour TSO                                                                   │ Exemple concret sur TSO                                                                           │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ edit-document       │ Améliorer paper.md — restructurer, clarifier, resserrer le style                  │ ➜ Le paper.md fait 55 pages → edit-document propose une table des matières, des titres plus         │
│                     │                                                                                  │   clairs, réduit les répétitions entre §3 et §5, ajoute des renvois                                │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ write-document      │ Créer des specs architecturales solides dans specs/                               │ ➜ Produit specs/tech-architecture/TECH_STACK.md pour TSO : dépendances, modules, flux de données,   │
│                     │                                                                                  │   diagramme d'architecture, décisions clés                                                         │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ commit-message      │ Générer des messages de commit Conventional Commits (semver-aware)                │ ➜ git diff → commit-message propose :                                                              │
│                     │                                                                                  │   « fix(core): resolve Φ oscillation deadlock in resolve_anneal » → semver: patch                   │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ elaborate-spec      │ Transformer une idée vague en spécification détaillée par dialogue                │ ➜ Tu : « J'aimerais que l'agent ait de la mémoire procédurale » → dialogue jusqu'à une spec        │
│                     │                                                                                  │   complète : stockage, rappel, coût métabolique, interaction avec le Cervelet                      │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ research-first      │ Chercher les solutions existantes avant d'implémenter (anti re-invention)          │ ➜ Tu veux implémenter un module d'attention bottom-up → research-first cherche les papiers         │
│                     │                                                                                  │   récents, les crates Rust existantes, des implémentations de référence → note « Prior Art »        │
├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ trace-requirement   │ Lier les stories du plan aux fichiers de code et de test (traçabilité)             │ ➜ Après une release : trace-requirement lit release-plan.yaml, épiques, et trouve les fichiers      │
│                     │                                                                                  │   de code/test qui implémentent chaque story → produit specs/TRACEABILITY_LATEST.md               │
└─────────────────────┴──────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Exécution & Delivery

```
┌──────────────────────┬───────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill                │ Utile pour TSO                                                                    │ Exemple concret sur TSO                                                                           │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ execute-plan         │ Exécuter les tâches d'un epic capsule pas à pas avec checkpoint humain             │ ➜ On a un epic « Ajouter le module d'attention Bayésienne » avec 5 stories → execute-plan les      │
│                      │                                                                                   │   exécute une par une, s'arrête après chaque pour que tu valides                                   │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ build-epic           │ Cycle complet en 8 étapes pour livrer un epic (lecture → plan → build → vérifie)   │ ➜ Epic « Nouveau mécanisme de sommeil avec consolidation hippocampique » → build-epic traverse     │
│                      │                                                                                   │   8 étapes : état des lieux, plan, implé, tests, review, merge                                      │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ delegate-task        │ Confier une tâche complexe à un subagent avec review en 2 temps                     │ ➜ « Implémente la version parallèle de resolve_anneal » → un subagent code, un reviewer            │
│                      │                                                                                   │   critique, toi tu valides et merges                                                               │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ dispatch-agents      │ Lancer plusieurs subagents en parallèle sur des tâches indépendantes               │ ➜ En parallèle : agent A écrit les tests du graphe sémantique, agent B refactore core.rs,          │
│                      │                                                                                   │   agent C met à jour paper.md → 3x plus vite                                                      │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ compose-workflow     │ Chaîner plusieurs skills dans un workflow custom pour TSO                          │ ➜ Workflow custom « Nouveau module TSO » : survey-context → define-language → spike-prototype      │
│                      │                                                                                   │   → plan-work → develop-tdd → verify-work → audit-code → release-branch                           │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ verify-work          │ UAT multi-phase : build → typecheck → lint → tests → vérification manuelle         │ ➜ Après l'implémentation du Démineur : cargo build, cargo check, cargo clippy, cargo test,         │
│                      │                                                                                   │   lance le binaire weakness_game, vérifie que PROOF SCORE = 100.0                                  │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ security-review      │ Analyse de sécurité des changements (injection, auth bypass, secrets)               │ ➜ Scanne le code pour : paths non sanitizés, unwrap() sur des entrées utilisateur,                 │
│                      │                                                                                   │   fuite de seed aléatoire, données de modèle sérialisées sans validation                          │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ publish-package      │ Publier sur npm / crates.io / PyPI si TSO devient une bibliothèque                 │ ➜ tso-engine devient une crate Rust → publish-package vérifie Cargo.toml, cargo publish --dry-run, │
│                      │                                                                                   │   puis cargo publish                                                                               │
└──────────────────────┴───────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Méta / Organisation

```
┌──────────────────────┬───────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Skill                │ Utile pour TSO                                                                    │ Exemple concret sur TSO                                                                           │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ orchestrate-project  │ Meta-skill qui coordonne les 6 phases (discover → elaborate → plan → build →      │ ➜ Tu lances orchestrate-project une fois → il te guide phase par phase : d'abord DISCOVER           │
│                      │ verify → release) avec des gates dures                                            │   (survey-context), puis ELABORATE (define-language), puis PLAN… un gate vérifie chaque phase       │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ seed-conventions     │ Générer CLAUDE.md + CONVENTIONS.md + specs/ pour un nouveau projet                │ ➜ Si t'avais commencé TSO sans specs : seed-conventions crée CLAUDE.md (stack Rust, commandes),     │
│                      │                                                                                   │   CONVENTIONS.md (TDD, coding rules), dossier specs/                                                │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ organize-workspace   │ Nettoyer les artifacts temporaires, caches, drafts, .gitignore — « clean my room » │ ➜ Trouve les fichiers .bin, .DS_Store, __MACOSX, les vieux binaires dans target/…                  │
│                      │                                                                                   │   → propose de les gitignorer, nettoyer, réorganiser                                               │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ plan-refactor        │ Planifier un refactoring en petits commits sûrs via interview                      │ ➜ Tu veux refactorer l'AttractorField (trop de responsabilités) : plan-refactor te guide pour      │
│                      │                                                                                   │   découper en 3 commits sûrs, chacun vérifiable                                                    │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ change-request       │ Ajouter un requirement ou réordonner les epics par WSJF en cours de release        │ ➜ En pleine release, tu te rends compte que le module de « peur » est plus important que            │
│                      │                                                                                   │   « mémoire procédurale » → change-request le remonte dans la priorité WSJF                       │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ plan-release         │ Ordonnancer les epics dans release-plan.yaml avec WSJF + BCP baselines             │ ➜ Tu as 5 epics (sommeil v2, attention bayésienne, mémoire procédurale, démineur v2,               │
│                      │                                                                                   │   logging) → plan-release les ordonnance par WSJF avec date cible                                  │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ find-way             │ Planifier un gros effort via des tickets de décision sur l'issue tracker            │ ➜ Tu veux faire de TSO une plateforme reproductible de recherche en neuroscience computationnelle   │
│                      │                                                                                   │   → find-way crée 20 tickets de décision, on les résout un par un                                  │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ visual-dashboard     │ Lancer un dashboard browser qui visualise l'architecture et le statut du projet    │ ➜ Lance un serveur HTTP local → dashboard web avec : graphe des modules TSO, barre de progression  │
│                      │                                                                                   │   des epics, courbe de Φ en temps réel                                                             │
├──────────────────────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ map-codebase         │ Dériver le tech-stack complet en scannant le code (quand specs/ n'existe pas)      │ ➜ Scanne tso-engine/src/ → produit specs/tech-architecture/tech-stack.md avec tous les modules,    │
│                      │                                                                                   │   dépendances externes (zod dans .opencode ???), binaires, structure des dossiers                  │
└──────────────────────┴───────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

> **💡 Pour lancer une skill :** il suffit de me dire son nom dans la conversation.  
> Par exemple : « lance **deepen-architecture** sur TSO », ou « fais un **survey-context** ».
>
> Fichier : `SKILLS_TSO_REFERENCE.md` — 7 sections, ~45 skills, une colonne exemple concret par skill
