## 8. Limites et travaux futurs

**Evaluation insuffisante.** Les benchmarks actuels (Terrarium 7x7,
Rotating-T 5x5, GridWorld 5x5) restent de petite taille et ne
couvrent pas les environnements a fort aliasing ou grande echelle.
Les resultats de section 6.7 (TSO + VAE sur entree 25D) sont prometteurs
mais ne remplacent pas une evaluation sur MiniGrid ou Procgen avec
observations visuelles reelles, 10+ seeds, intervalles de confiance.
Le faible nombre de seeds (5-30 selon les experiences) limite la
generalisation statistique des resultats.

**La tension cognitive Phi n'est pas validee en aliasing severe.**
La preuve de concept de Phi comme mecanisme de detection de conflit
est etablie sur grilles 5x5 (section 5), mais son apport sur des POMDP
visuels complexes n'est pas mesure. Une experience sur MiniGrid avec
observations partielles (ex: MiniGrid-DoorKey-5x5-v0) ou l'aliasing
est structurel et non positionnel est necessaire pour valider que Phi
resolve de vrais problemes d'ambiguite perceptuelle.

**Pistes pour la suite :**
1. Benchmark MiniGrid (observations visuelles 7x7x3, VAE vers 16D)
   avec 10 seeds, intervalles de confiance, ablation de chaque
   sous-systeme (VAE, attracteur, episodique, Phi).
2. Passage a Procgen (environnements 64x64) via le bridge PyO3
   existant (tso_env), avec evaluation systematique sur les
   16 jeux de Procgen.
3. Analyse de sensibilite complete de Phi sur POMDP : est-ce que le
   graphe semantique detecte les changements de contexte mieux qu'une
   ligne de base avec simple memoire de travail (fenetre de contexte) ?
4. Release d'un benchmark standardise tso-bench avec seeds fixes,
   intervalles de confiance, et scripts de reproduction automatiques.