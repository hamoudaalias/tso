"""
TSO NLP Layer — language interface built on PyTorch + HuggingFace.

Depends on tso_kernel for the core engine, adds embeddings,
tokenization, SOM clustering, and the conceptual decoder.
"""
from .embedder import MiniLMEmbedder
from .som import SOM
from .decoder import ConceptualDecoder, InverseMotor
