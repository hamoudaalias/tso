import numpy as np
from collections import defaultdict, Counter
from itertools import combinations


class HierarchicalGraph:
    """
    Trois niveaux topologiques superposés :

    - α (Lexical)     : graphe de co-occurrences temporelles (R-STDP)
    - β (Compositionnel) : nœuds fusion recrutés sur paires α fréquentes
    - γ (Conceptuel)  : tri-friction [support, conflict, novelty]

    Optimisations :
    - Vocabulaire plafonné (max_vocab), tokens rares → <unk>
    - Élagage périodique des arêtes faibles
    - Voisinage incrémental (pas de scan O(E) dans _neighbors)
    """

    def __init__(
        self,
        window_size=5,
        fusion_threshold=0.7,
        top_k=20,
        max_beta=2048,
        max_vocab=10000,
        cleanup_interval=1000,
        min_edge_weight=0.1,
    ):
        self.window_size = window_size
        self.fusion_threshold = fusion_threshold
        self.top_k = top_k
        self.max_beta = max_beta

        # Optimisation 1 : vocabulaire limité
        self.max_vocab = max_vocab
        self.token_counter = Counter()
        self.vocab = set()
        self.vocab_locked = False
        self.unk_token = "<unk>"

        # Optimisation 2 : élagage périodique
        self.cleanup_interval = cleanup_interval
        self.min_edge_weight = min_edge_weight
        self.sentence_count = 0

        # α — voisinage incrémental (Optimisation 3)
        # neighbors[word] = {neighbor: weight}
        self.neighbors = defaultdict(dict)
        self.alpha_nodes = set()
        self.eligibility = defaultdict(float)

        # β
        self.beta_nodes = {}
        self.beta_nodes_rev = {}

        # Ph27 : friction interne inter-fenêtres
        self.last_phi_context = 0.0

        # état courant
        self.active_alpha = []
        self.active_beta = set()

    # ── vocable ──────────────────────────────────────────────────

    def _resolve_token(self, token):
        if not self.vocab_locked:
            self.token_counter[token] += 1
            if len(self.vocab) < self.max_vocab:
                self.vocab.add(token)
                return token
            # verrouiller le vocab une fois plein
            self.vocab_locked = True
            # garder les max_vocab plus fréquents
            most_common = {t for t, _ in self.token_counter.most_common(self.max_vocab)}
            self.vocab = most_common
            return token if token in most_common else self.unk_token
        return token if token in self.vocab else self.unk_token

    # ── α : R-STDP ──────────────────────────────────────────────

    def process_token(self, raw_token):
        token = self._resolve_token(raw_token)
        self.alpha_nodes.add(token)

        for prev in self.active_alpha[-self.window_size:]:
            if prev == token:
                continue
            edge = (prev, token)
            # mise à jour incrémentale du voisinage
            old_w = self.neighbors[prev].get(token, 0.0)
            new_w = old_w + 0.1
            self.neighbors[prev][token] = new_w
            self.neighbors[token][prev] = new_w
            self.eligibility[edge] += 0.05

        self.active_alpha.append(token)
        if len(self.active_alpha) > self.window_size:
            self.active_alpha.pop(0)

        self._maybe_fuse()
        self._activate_beta()

    # ── β : fusion compositionnelle ─────────────────────────────

    def _maybe_fuse(self):
        active = list(set(self.active_alpha))
        for a, b in combinations(active, 2):
            edge = (a, b)
            if self.eligibility[edge] <= self.fusion_threshold:
                continue
            key = (a, b)
            if key not in self.beta_nodes_rev:
                if len(self.beta_nodes) >= self.max_beta:
                    continue
                bid = len(self.beta_nodes)
                self.beta_nodes[bid] = key
                self.beta_nodes_rev[key] = bid
            self.eligibility[edge] = 0.0

    def _activate_beta(self):
        self.active_beta.clear()
        cur = set(self.active_alpha)
        for bid, (w1, w2) in self.beta_nodes.items():
            if w1 in cur and w2 in cur:
                self.active_beta.add(bid)

    # ── Ph28 : recrutement β forcé par conflit γ ────────────────

    def force_beta_node(self, left_tokens, right_tokens):
        """Crée un nœud β à partir de la friction cross-window.
        Utilise un token représentatif de chaque moitié comme clé."""
        def pick_rep(seq):
            for t in seq:
                if t != self.unk_token and t in self.vocab:
                    return t
            return seq[0] if seq else self.unk_token
        rep_left = pick_rep(left_tokens)
        rep_right = pick_rep(right_tokens)
        if rep_left == rep_right:
            return -1
        key = tuple(sorted((rep_left, rep_right)))
        if key not in self.beta_nodes_rev:
            if len(self.beta_nodes) >= self.max_beta:
                # β saturé, on remplace un nœud aléatoire
                victim = np.random.randint(0, self.max_beta)
                old_key = self.beta_nodes[victim]
                del self.beta_nodes_rev[old_key]
                self.beta_nodes[victim] = key
                self.beta_nodes_rev[key] = victim
                return victim
            bid = len(self.beta_nodes)
            self.beta_nodes[bid] = key
            self.beta_nodes_rev[key] = bid
            return bid
        return self.beta_nodes_rev[key]

    # ── γ : tri-friction (voisinage O(1) par mot) ───────────────

    def _neighbors(self, word):
        """Retourne les top_k voisins de `word` via le dictionnaire incrémental."""
        if word not in self.neighbors:
            return set()
        sorted_n = sorted(self.neighbors[word].items(), key=lambda x: -x[1])
        return {n for n, _ in sorted_n[:self.top_k]}

    def compute_tri_friction(self, premise_tokens, hypothesis_tokens):
        N_premise = set()
        for w in premise_tokens:
            N_premise |= self._neighbors(w)
        N_hypothesis = set()
        for w in hypothesis_tokens:
            N_hypothesis |= self._neighbors(w)

        inter = N_premise & N_hypothesis
        union = N_premise | N_hypothesis
        support = len(inter) / len(union) if union else 0.0

        conflict = len(N_premise - N_hypothesis) / len(N_premise) if N_premise else 0.0

        novel = (
            sum(1 for w in hypothesis_tokens if w not in N_premise)
            / len(hypothesis_tokens)
            if hypothesis_tokens
            else 0.0
        )

        return np.array([support, conflict, novel], dtype=np.float32)

    # ── élagage périodique ──────────────────────────────────────

    def _prune_edges(self):
        """Supprime les arêtes faibles et les entrées d'éligibilité orphelines."""
        to_remove = []
        for word in list(self.neighbors.keys()):
            for nb, w in list(self.neighbors[word].items()):
                if w < self.min_edge_weight:
                    to_remove.append((word, nb))
        for a, b in to_remove:
            if b in self.neighbors.get(a, {}):
                del self.neighbors[a][b]
            if a in self.neighbors.get(b, {}):
                del self.neighbors[b][a]
        # nettoyage éligibilité
        stale = [k for k, v in self.eligibility.items() if v < self.min_edge_weight * 0.5]
        for k in stale:
            del self.eligibility[k]
        # nettoyage nœuds orphelins
        empty = [w for w, n in self.neighbors.items() if not n]
        for w in empty:
            del self.neighbors[w]

    # ── γ : friction interne inter-fenêtres (Ph27) ──────────────

    def compute_cross_window_friction(self, tokens):
        """Découpe la phrase en deux moitiés, calcule la tri-friction entre elles."""
        n = len(tokens)
        if n < 2:
            return np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mid = n // 2
        left = tokens[:mid]
        right = tokens[mid:]
        gamma = self.compute_tri_friction(left, right)
        self.last_phi_context = float(gamma[1])  # conflict
        return gamma

    # ── état topologique s ──────────────────────────────────────

    def process_sentence(self, sentence_tokens):
        """Traite tous les tokens d'une phrase (appelé une seule fois)."""
        self.active_alpha.clear()
        self.active_beta.clear()
        for t in sentence_tokens:
            self.process_token(t)
        self.sentence_count += 1
        if self.sentence_count % self.cleanup_interval == 0:
            self._prune_edges()

    def build_state_from_active(self, sentence_tokens):
        """Construit le vecteur d'état s à partir de l'état actif (sans reprocessing)."""
        beta_vec = np.zeros(self.max_beta, dtype=np.float32)
        for bid in self.active_beta:
            beta_vec[bid] = 1.0

        gamma_vec = self.compute_cross_window_friction(sentence_tokens)

        return np.concatenate([beta_vec, gamma_vec])

    def get_topological_state(self, sentence_tokens):
        """Traite la phrase ET construit l'état (usage externe, une passe)."""
        self.process_sentence(sentence_tokens)
        return self.build_state_from_active(sentence_tokens)
