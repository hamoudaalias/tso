// Module Neurogenesis — Sommeil Phase 3
// Design 1 (Minimal) pour l'interface publique, Design 2 (Phases) pour les tests.
// Voir specs/tech-architecture/neurogenesis-interface-design.md

use ndarray::Array1;
use rand::Rng;
use crate::attractor::AttractorField;
use crate::core::Graph;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration complète de la neurogenèse.
/// 4 champs — tout le comportement est là.
#[derive(Clone, Debug)]
pub struct NeurogenesisConfig {
    /// Probabilité qu'un concept existant génère un nouveau (0.0 – 1.0)
    pub rate: f64,
    /// Nombre maximum de concepts (prototypes + graphe)
    pub max_concepts: usize,
    /// Cycles sommeil de période critique pour les nouveau-nés
    pub maturation_cycles: usize,
    /// Scaling synaptique des arêtes après neurogenèse
    pub synaptic_scaling: bool,
}

impl Default for NeurogenesisConfig {
    fn default() -> Self {
        NeurogenesisConfig {
            rate: 0.2,
            max_concepts: 50,
            maturation_cycles: 3,
            synaptic_scaling: true,
        }
    }
}

// ── Résultat ────────────────────────────────────────────────────────────

/// Ce que le caller reçoit après un cycle de neurogenèse.
#[derive(Clone, Debug)]
pub struct NeurogenesisOutcome {
    pub births: usize,
    pub deaths: usize,
}

// ── Module principal ────────────────────────────────────────────────────

/// Module de neurogenèse structurelle.
/// Cycle de vie : birth → maturation → homeostatic replacement → synaptic scaling.
pub struct Neurogenesis {
    pub config: NeurogenesisConfig,
    /// Compteur de maturation pour chaque concept (0 = mature, >0 = période critique).
    /// Indexé par concept_id. Géré en interne, synchronisé avec TsoEngine via cycle().
    pub(crate) maturation: Vec<usize>,
}

impl Neurogenesis {
    pub fn new(config: NeurogenesisConfig) -> Self {
        Neurogenesis {
            config,
            maturation: Vec::new(),
        }
    }

    /// Méthode publique unique — tout arrive ici.
    /// Retourne un outcome + le vecteur de maturation mis à jour.
    pub fn cycle(
        &mut self,
        attractor: &mut AttractorField,
        graph: &mut Graph,
        last_active_step: &[usize],
        noise_std: f64,
    ) -> NeurogenesisOutcome {
        // Synchroniser maturation avec le nombre actuel de concepts
        while self.maturation.len() < attractor.n_classes() {
            self.maturation.push(0);
        }

        let births = self.birth_phase(attractor, graph, last_active_step, noise_std);
        self.homeostasis(attractor, graph, last_active_step);
        if self.config.synaptic_scaling {
            self.scale_synapses(graph);
        }
        self.end_cycle();
        NeurogenesisOutcome { births, deaths: 0 }
    }

    // ── Phase 1 : Naissance de nouveaux concepts ──

    fn birth_phase(
        &mut self,
        attractor: &mut AttractorField,
        graph: &mut Graph,
        _last_active_step: &[usize],
        noise_std: f64,
    ) -> usize {
        if self.config.rate <= 0.0 || self.config.max_concepts == 0 {
            return 0;
        }

        let mut new_concepts = 0usize;
        let n_classes = attractor.n_classes();

        for i in 0..n_classes {
            if attractor.n_classes() >= self.config.max_concepts {
                break;
            }
            if rand::random::<f64>() >= self.config.rate {
                continue;
            }
            if let Some(proto) = attractor.get_prototype(i) {
                let noise: Array1<f64> = (0..proto.len())
                    .map(|_| rand::random::<f64>() * noise_std * 2.0 - noise_std)
                    .collect();
                let mutated = proto + &noise;

                let new_id = attractor.add_class(&mutated);

                // Ajouter un nœud dans le graphe avec le prototype comme embedding
                let node_embedding = attractor.get_prototype(new_id)
                    .cloned()
                    .unwrap_or_else(|| mutated.clone());
                let node_idx = graph.add_node(node_embedding);

                // Connecter aléatoirement à 2-3 voisins
                let graph_nodes = graph.nodes.len();
                if graph_nodes > 1 {
                    let n_edges = std::cmp::min(rand::random::<usize>() % 2 + 2, graph_nodes - 1);
                    for _ in 0..n_edges {
                        let neighbor = rand::random::<usize>() % (graph_nodes - 1);
                        if neighbor != node_idx {
                            graph.add_edge(node_idx, neighbor, 1);
                        }
                    }
                }

                // Initialiser la période critique
                while self.maturation.len() <= node_idx {
                    self.maturation.push(0);
                }
                self.maturation[node_idx] = self.config.maturation_cycles;
                new_concepts += 1;
            }
        }
        new_concepts
    }

    /// Trouve le concept le moins actif (hors période critique) pour remplacement.
    fn find_least_active_concept(&self, last_active_step: &[usize]) -> Option<usize> {
        let n = last_active_step.len();
        if n == 0 {
            return None;
        }
        (0..n)
            .filter(|&i| self.maturation.get(i).copied().unwrap_or(0) == 0)
            .min_by_key(|&i| last_active_step.get(i).copied().unwrap_or(0))
    }

    // ── Phase 2 : Homéostasie — remplacer si budget plein ──

    fn homeostasis(
        &mut self,
        attractor: &mut AttractorField,
        _graph: &mut Graph,
        last_active_step: &[usize],
    ) {
        while attractor.n_classes() >= self.config.max_concepts {
            let target = match self.find_least_active_concept(last_active_step) {
                Some(t) => t,
                None => break, // tous en période critique
            };

            // Forcer le retrait du concept cible via l'attractor
            // En pratique, le caller (sleep_cycle) appelle prune_concepts après cycle()
            // qui se charge du nettoyage. Ici on se contente de marquer la cible.
            // TODO: mécanisme de remplacement direct dans attractor
            break;
        }
    }

    // ── Phase 3 : Scaling synaptique ──

    fn scale_synapses(&self, graph: &mut Graph) {
        let n_nodes = graph.nodes.len();
        if n_nodes == 0 {
            return;
        }
        let mut incident_weight = vec![0i64; n_nodes];
        for e in &graph.edges {
            incident_weight[e.from] += e.weight as i64;
            incident_weight[e.to] += e.weight as i64;
        }
        let mean_weight = incident_weight.iter().sum::<i64>() as f64 / n_nodes as f64;
        let threshold = mean_weight * 2.0;
        if threshold <= 0.0 {
            return;
        }
        for (i, &total) in incident_weight.iter().enumerate() {
            let total_f = total as f64;
            if total_f > threshold {
                let scale = threshold / total_f;
                for e in &mut graph.edges {
                    if e.from == i || e.to == i {
                        let scaled = (e.weight as f64 * scale).round() as i8;
                        e.weight = scaled.clamp(0, 127);
                    }
                }
            }
        }
    }

    /// Fin de cycle : décrémenter les compteurs de maturation
    fn end_cycle(&mut self) {
        for m in &mut self.maturation {
            if *m > 0 {
                *m -= 1;
            }
        }
    }

    /// Retourne l'état de maturation (debug/inspection)
    pub fn maturation_snapshot(&self) -> &[usize] {
        &self.maturation
    }
}
