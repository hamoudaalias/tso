"""
TSO-MiniLM — Auto-encodeur topologique zero-gradient.

Produit un embedding 384-d à partir de la dynamique interne des
trois niveaux α, β, γ, sans aucune supervision externe.
"""
from .hierarchy import HierarchicalGraph
from .routing import FrictionRouter
from .autoencoder import TSO_MiniLM_Autoencoder
