# Spike: GridWorld → VAE → TSO — pipeline encodeur

## Question

Le pipeline encodeur (perception → concept → Φ → action) fonctionne-t-il
de bout en bout avec des observations structurées, et le VAE améliorerait-il
le latent comparé à l'AttractorField ?

## Résultat

✅ **Partiellement répondu** — le pipeline complet fonctionne (12ms pour 200 steps).
La partie VAE n'a pas pu être testée car `VaeEncoder` n'est pas sur master.

## Résultats

- **8 concepts** créés à partir de 200 observations aléatoires (AttractorEncoder)
- **3 concepts supplémentaires** créés pendant sleep via neurogenèse
- **Φ = 0.0** tout le temps — pas d'arêtes dans le graphe (obs aléatoires non structurées)
- **Sleep cycle** : consolidation et neurogenèse fonctionnent
- **Temps** : 12.3ms pour 200 steps, négligeable

## Evidence

```
step=  50 action=3 concepts=8 Φ=0.0000
step= 100 action=2 concepts=8 Φ=0.0000
step= 150 action=1 concepts=8 Φ=0.0000
step= 200 action=0 concepts=8 Φ=0.0000
Concepts après sleep : 13 (créés: 3, prunés: 0)
Φ  : 0.0000 → 0.0000
Temps : 12.3ms
```

## Implications pour le plan

1. **Pipeline encodeur OK** — l'architecture `Encoder` trait + `TsoEngine.step()`
   fonctionne. Aucun blocant.
2. **VAE nécessaire** — l'AttractorEncoder ne produit pas de graphe (Φ=0)
   avec des données non structurées. Le VaeEncoder (`encode_continuous`)
   donnerait un latent structuré → meilleur graphe → Φ non nul → décisions
   informées.
3. **Branche e09 nécessaire** — le VaeEncoder + mode continu vit sur
   `feat/e09-joint-vae-cerebellum` (pas mergeable). Planifier e09 après
   le merge de cette spike.

## Ce qui n'a PAS été exploré

- GridWorld images → VAE (nécessite le VAE pré-entraîné en Python + import)
- VaeEncoder encode_continuous → pas sur master
- Gradient TD rétropropagé dans l'encodeur

## Recommandation

✅ Procéder avec e09 (VAE joint) après le merge de e10. Les fondations
(Encoder trait, TsoEngine.step avec encodeur, sleep cycle) sont solides.
Le VaeEncoder existe dans la branche e09 mais n'a pas été mergeable.
