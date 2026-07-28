# VAE Vision 8×8 — Livrable

## Résultat
**Encoder trait + VaeEncoder pour images 8×8** fonctionnel avec les
limitations documentées (entraînement en ligne = instabilité catégorielle).

## Métriques

| Métrique | Valeur |
|----------|--------|
| Architecture | 64→32→8 (latent)→32→64 |
| Catégories (200 steps) | 368 |
| MSE reconstruction | 0.336 (min 0.058, max 0.713) |
| Stabilité (50× même image) | 50 catégories distinctes (2% majoritaire) |
| Types d'images regroupés | 0/4 variants dans la même catégorie |

## État

- ✅ **Encoder trait** : encode_raw() → EncodeResult (category_id, novelty, is_new)
- ✅ **VaeEncoder** : VAE complet (encode → reparameterize → decode → centroid)
- ✅ **Intégration TsoEngine** : `self.encoder` remplace l'attractor dans step()
- ✅ **Images 8×8** : input 64D, latent 8D, forward pass valide
- ✅ **Reconstruction** : MSE moyenne 0.33 (bonne qualité visuelle)
- ❌ **Stabilité** : catastrophique (centroid explosion) — voir spike
- ❌ **Entraînement** : couche finale uniquement, encodeur ne converge pas

## Commande

```bash
cd tso-engine && cargo run --bin livrable_vae_vision
```

## Architecture

```
┌──────────┐   ┌──────────────────────────────────────┐   ┌──────────┐
│ Image     │──→│ VaeEncoder (64→32→8→32→64)          │──→│ category │
│ 8×8 (64D) │   │  encode → z → centroid → best match │   │  id      │
└──────────┘   └──────────────────────────────────────┘   └──────────┘
                     │
                     ├── VaeStats: mu, logvar, ELBO, KL, MSE
                     │
                     └── prototype: None (VaeEncoder ne stocke pas de prototypes)
```
