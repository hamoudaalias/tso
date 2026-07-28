# Spike: VAE encodeur sur TSO GridWorld 5×5

## Question
Le VAE peut-il remplacer l'AttractorField dans le cycle cognitif TSO en ligne
(1 step = 1 encode + 1 train_step) ? Autrement dit : l'encodage variationnel
appris pas-à-pas converge-t-il assez vite pour alimenter le graphe sémantique,
la mémoire épisodique, et l'apprentissage par renforcement ?

## Result
**Non, pas en l'état.** Le VAE en ligne (sans mini-batchs, sans réentraînement
profond) est instable et inutilisable comme remplacement direct de l'AttractorField.

## Findings

### 1. Explosion catégorielle : 207 catégories en 100 épisodes
L'AttractorField crée typiquement 5-15 concepts pour le GridWorld 5×5. Le VAE
en crée **207**. Cause : chaque perception légèrement différente produit un
latent différent qui tombe au-delà du seuil du centroid le plus proche → nouvelle
catégorie à chaque step.

### 2. Instabilité catastrophique : 78 catégories pour la même entrée
En présentant 100 fois exactement la même perception, le VAE la mappe dans
**78 catégories différentes**. À chaque appel, la stochasticité du
reparametrization trick (`z = µ + σ·ε`) produit un latent différent, et le
centroid le plus proche n'est jamais le même car les centroids bougent aussi
entre deux appels.

### 3. MSE de reconstruction stable mais élevé (~0.227)
Le VAE reconstruit correctement (MSE ~0.23 sur des entrées en [0,1]),
ce qui montre que l'encodeur/décodeur apprend. Mais c'est la **sortie latente**
qui est instable, pas la reconstruction.

### 4. KL divergence très faible (~0.0104)
KL = 0.01 avec 3 dimensions latentes. En pratique, le VAE ignore la régularisation
KL et fonctionne comme un auto-encodeur vanilla. Les latents ne sont pas
structurellement organisés.

### 5. Entraînement en ligne insuffisant
Le `train_step` actuel ne met à jour que la dernière couche (`w_dec`, `b_dec`).
Les poids de l'encodeur (`w_enc`, `w_mu`, `w_logvar`) ne bougent jamais.
En conséquence, l'encodeur reste aléatoire et ne peut pas apprendre des
représentations latentes utiles.

## Evidence
```text
Épisodes : 100, Temps : 332.8ms
Catégories finales : 207 (vs 5-15 pour AttractorField)
Moyenne MSE : 0.227, Moyenne KL : 0.0104
Stabilité : 78 catégories pour 100× la même perception
```

## Implications for the plan
1. **Ne pas remplacer AttractorField par le VAE actuel en ligne.** L'instabilité
   catastrophique des catégories détruirait la mémoire épisodique (qui stocke
   des séquences de category_id) et le graphe sémantique (qui accumule des
   arêtes entre catégories).
2. **Pour la vision (pixels → latent), le VAE doit être pré-entraîné hors ligne**
   sur un dataset fixe, puis utilisé en inférence seule dans le cycle TSO.
3. **L'entraînement complet (encodeur + décodeur)** nécessite une
   rétropropagation totale (backprop through tanh + linear layers), pas
   seulement la dernière couche.

## What was NOT explored
- VAE pré-entraîné hors ligne (batch training sur dataset fixe) puis utilisé
  en inférence seule dans TSO
- Backpropagation complète avec autograd manuel ou mini-batchs
- VAE convolutionnel pour entrées image (nécessite vision, pas moustaches)
- Variation du seuil de centroids (>1.0) pour réduire l'explosion catégorielle
- Couplage VAE + buffer de rejeu (rejouer les transitions pour stabiliser
  l'apprentissage en ligne)

## Recommendation
**Construire un auto-encodeur pré-entraîné** comme encodeur de vision (pixels
d'une caméra → latent 16-64D → category_id stable). Le VAE en ligne strict
(1 step = 1 update) ne remplace pas l'AttractorField. Utiliser l'AttractorField
pour les moustaches (catégories stables et parcimonieuses), et réserver le VAE
pour les entrées continues de haute dimension (vision) après pré-entraînement
hors ligne.
