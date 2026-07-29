# Model Domain — Stress Test du papier contre le domaine TSO

## Décision 1 : "VAE online" vs réalité ADR

**ADR-001-encoder-classifier-separation** (accepté) dit :
> "Le VAE existant nécessite un pré-entraînement hors-ligne"

Le papier dit "VAE online + Gumbel-STE". **Contradiction.**

**Décision :** Renommer en "VAE avec Gumbel-STE" sans "online".
Si l'inférence est en ligne mais l'entraînement hors-ligne, le dire.

## Décision 2 : Gradient coupé VAE↔Cerebellum — non documenté

Le tech-stack.md dit :
> "Le VAE et le cervelet sont deux branches parallèles qui ne communiquent
> pas. Le cervelet apprend sur la perception brute, pas sur le latent.
> Le gradient est coupé à l'assignation discrète des centroids."

Le papier ne mentionne pas cette coupure. Un reviewer qui lit le code
verra que le papier ment par omission.

**Décision :** Ajouter une phrase explicite : "Le gradient TD ne remonte
pas dans le VAE ; l'apprentissage des prototypes est déconnecté du
signal de récompense."

## Décision 3 : Φ gating — pas implémenté dans la version benchmarkée ?

Le papier décrit phi_gating comme une étape 4 du heartbeat, mais les
résultats du benchmark n'isolent pas son effet. Est-il activé ?

**Décision :** Soit (a) mesurer l'économie de calcul avec/sans gating,
soit (b) dire explicitement "phi_gating n'est pas isolé dans le
benchmark actuel."

## Décision 4 : Catégorisation par prototypes — unique ?

L'UBIQUITOUS LANGUAGE mentionne "AttractorField" comme mécanisme de
catégorisation. Trois implémentations existent (attracteur, VAE, FPI).
Le papier ne compare pas ces 3 approches entre elles.

**Décision :** Ajouter une micro-comparaison attracteur vs VAE vs FPI
sur le même benchmark.

## Décision 5 : Reward non-stationnaire — contrib ou détail ?

L'ADR-006 (récompense non-stationnaire) documente que le benchmark
change de but tous les 50 épisodes. C'est un détail important qui
explique pourquoi un RL naïf échoue — mais le papier ne le mentionne
pas comme défi distinct.

**Décision :** Mentionner explicitement le changement de but et pourquoi
cela favorise une architecture avec mémoire/Φ.

## Verdict final du model-domain

Le papier fait 5 écarts avec le domaine réel documenté dans les ADR
et le tech-stack. Chaque écart est corrigible : il faut aligner le
texte sur ce qui existe vraiment, pas sur ce qui serait idéal.
