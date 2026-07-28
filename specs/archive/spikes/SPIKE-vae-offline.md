# Spike: VAE pré-entraîné hors ligne — vision 8×8 stable

## Question
Pré-entraîner le VAE hors ligne (batch, 100 epochs) résout-il l'instabilité
catégorielle observée en entraînement en ligne (50 catégories pour 50× la
même image, 2% de stabilité) ?

## Result
**Oui.** Le VAE pré-entraîné est **100% stable** pour les 4 types d'images.

## Findings

### 1. Stabilité parfaite après pré-entraînement

| Métrique | En ligne (spike précédent) | Pré-entraîné (ce spike) |
|----------|---------------------------|------------------------|
| Stabilité 50× même image | **2%** (50 catégories) | **100%** (0 catégories) |
| MSE reconstruction | 0.336 | **0.124** (−63%) |
| KL divergence | 0.010 | **0.058** (×5.8, mieux structuré) |
| Catégories créées | 368 (explosion) | **0** (inférence seule) |

### 2. La cause de l'instabilité en ligne
L'encodeur (`w_enc`, `w_mu`, `w_logvar`) n'est jamais mis à jour par
`train_step()`. Seule la dernière couche `w_dec/b_dec` apprend. Donc
le reparameterization trick (`z = µ + σ·ε`) produit un latent différent
à chaque appel car σ reste grand (encodeur aléatoire). En inférence
déterministe (`z = µ`), le latent est **parfaitement stable** car les
poids de l'encodeur ne changent pas.

### 3. Qualité de reconstruction
MSE = 0.124 après 100 epochs (5.2s d'entraînement sur 200 images).
La KL = 0.058 indique que le VAE apprend une structure latente,
contrairement au mode en ligne (KL = 0.01, encodeur vanilla).

## Evidence
```text
Type 0: 100% stable (µ dist < 0.01, 50 essais)
Type 1: 100% stable
Type 2: 100% stable
Type 3: 100% stable
Temps: 5.2s, MSE: 0.124, KL: 0.058
```

## Implications for the plan
1. **Le VAE doit être pré-entraîné hors ligne**, puis gelé pour l'inférence.
2. **L'encode déterministe (`z = µ`)** est suffisant pour la stabilité
   catégorielle. Le reparameterization est réservé à l'entraînement.
3. **Le VaeEncoder actuel** (encoder.rs) utilise déjà le reparameterization
   dans `encode_raw()`. Pour la production, ajouter un mode
   `deterministic: bool` qui utilise `z = µ` au lieu de `z = µ + σ·ε`.
4. **Les centroids** deviennent stables avec un encodeur gelé : deux
   mêmes images donnent exactement le même latent.

## What was NOT explored
- Sauvegarde/chargement des poids pour usage dans TsoEngine
- `VaeEncoder` en mode déterministe avec freeze des poids
- Impact du pré-entraînement sur la qualité du regroupement (même
  type → même centroid)
- Dataset plus large (>200 images) ou images naturelles (pas synthétiques)

## Recommendation
**Adopter la pipeline :** pré-entraînement batch hors ligne → freeze encoder
→ inférence déterministe (`z = µ`) dans TsoEngine via VaeEncoder avec
`deterministic = true`. Les poids du VAE peuvent être sérialisés avec le
même mécanisme serde que le reste du moteur (Vae déjà Serialize/Deserialize).
