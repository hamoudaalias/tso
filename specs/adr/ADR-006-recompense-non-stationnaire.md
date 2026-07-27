# ADR-006 : Problème du signal de récompense non-stationnaire

## statut
Problème ouvert (investigation en cours)

## contexte
La Phase 1 a révélé un problème fondamental : le TSO complet (avec attracteur, graphe Φ, attention, well-being à 9 termes) obtient seulement 20% en exploitation pure sur un environnement 5×5 où le Cervelet seul obtient 98%.

## analyse
La cause racine est identifiée : le well-being dépend de l'état interne de l'agent (nombre de concepts, Φ, curiosité, homéostasie). Ces variables évoluent avec l'apprentissage, rendant le signal de récompense **non-stationnaire**. La cible TD du critic change constamment, empêchant la convergence.

## solutions tentées
| Solution | Résultat |
|----------|----------|
| Replay buffer (10 000 transitions) | ✅ Stabilise le Cervelet seul |
| Signal stationnaire (R_ext seul dans le replay) | ❌ Pas suffisant seul |
| Attention spatiale | À tester |

## décision
- Conserver le replay buffer (acquis)
- La non-stationnarité du well-being est désormais un **objet de recherche** du projet
- Pistes à explorer : normalisation du well-being, substract baseline roulante, séparation critic interne/externe

## conséquences
- + Le problème est clairement identifié et documenté
- + Le Cervelet seul fournit une baseline (98%) pour mesurer le progrès
- − Tant que ce problème n'est pas résolu, le TSO complet ne peut pas rivaliser avec le Cervelet seul
- − Nécessite des expériences supplémentaires (Phase 2)
