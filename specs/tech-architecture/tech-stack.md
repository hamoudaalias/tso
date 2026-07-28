# Deepen Architecture — Gradient joint VAE ↔ Cerebellum

## Flux actuel (gradient coupé)

```
perception (5D/6D)
    ├──→ VaeEncoder.encode_raw() → centroids → category_id → graphe/Φ
    └──→ decision_state (perception brute) → Cerebellum.forward_logits()
                                                  ↓
                                          reinforce_td()
                                                  ↓
                                     gradient TD dans w_lin / w1
                                     (jamais dans le VAE)
```

Le VAE et le cervelet sont **deux branches parallèles** qui ne communiquent pas. Le cervelet apprend sur la perception brute, pas sur le latent. Le VAE ne reçoit aucun signal RL. **Le gradient est coupé à l'assignation discrète des centroids.**

## Module Depth scores

| Module | Depth | Problème |
|--------|-------|----------|
| `VaeEncoder` (mode actuel) | 1 | Interface complexe (encode_raw → centroids → EncodeResult) pour zéro effet sur RL |
| `step()` decision_state | 1 | Passe la perception brute au cervelet, ignorant l'encodeur |
| `Encoder` trait | 5 | ✅ Bon — interchangeable |
| `Cerebellum::forward_logits` | 3 | Fonctionnel mais ne reçoit pas le bon input |

## Solution — Mode continu

### Seam 1 : `VaeEncoder::encode_continuous() → &[f64]`

Ajouter une méthode qui retourne le latent z (`µ` en mode déterministe) **sans passer par les centroids**. Interface :

```rust
impl VaeEncoder {
    /// Retourne le latent z (µ) directement — pas de centroids, pas de discrétisation.
    /// Le cervelet reçoit z comme état continu.
    pub fn encode_continuous(&mut self, perception: &Array1<f64>) -> &[f64] {
        // encode → mu (déterministe) ou z (stochastique si training)
        self.vae.encode(perception);
        if self.deterministic {
            &self.vae.mu
        } else {
            self.vae.reparameterize();
            &self.vae.z
        }
    }
}
```

**Tests qui survivent :** tous les tests existants sur encode_raw (centroids intacts).

### Seam 2 : `decision_state` = latent z si encodeur actif

Dans `step()`, remplacer :

```rust
// AVANT (gradient coupé)
let decision_state = if self.use_stationary_reward { perception.clone() } else { gated.clone() };

// APRÈS (gradient traverse l'encodeur)
let decision_state = if let Some(enc) = &mut self.encoder {
    if let Some(vae_enc) = enc.as_any().downcast_ref::<VaeEncoder>() {
        let z = vae_enc.encode_continuous(perception);
        Array1::from_vec(z.to_vec())
    } else {
        perception.clone()
    }
} else {
    perception.clone()
};
```

### Seam 3 : Gradient approximé VAE ← δ TD

Dans `reinforce_td` (ou après), ajouter :

```rust
// Gradient approximé : δ × ∂logits/∂z × ∂z/∂w_enc
// δ = TD_error (déjà calculé dans reinforce_td)
// ∂logits/∂z = w_lin (linéaire) ou w1 (MLP)
// ∂z/∂w_enc = h_enc[j] (tanh activations)
if let Some(vae_enc) = &mut self.vae_encoder {
    vae_enc.backprop_td(td_error, &cerebellum_gradient);
}
```

## Bénéfices

| Avant | Après |
|-------|-------|
| VAE et cervelet déconnectés | Gradient TD traverse le VAE |
| VAE gelé (freeze) = jamais entraîné par RL | VAE s'adapte à la tâche |
| Centroids figés → explosion ontologique | Latent continu → pas de centroids |
| Plafond 24% sur Minigrid | Potentiel >50% |

## Informations sur les dépendances

- `VaeEncoder` → `Vae` (in-process, pas d'I/O)
- `Cerebellum` → `VaeEncoder` (in-process, nouvelle dépendance dirigée)
- `TsoEngine::step()` → point d'intégration unique
- **Catégorie** : in-process pure computation — pas d'adapter nécessaire

## Prochaine étape

**design-interface** pour explorer 3 designs différents de l'interface encode_continuous + backprop, en parallèle.
