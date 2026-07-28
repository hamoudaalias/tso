// Module Neurogenesis — Sommeil Phase 3
// Design 1 (Minimal) pour l'interface publique, Design 2 (Phases) pour les tests.
// Voir specs/tech-architecture/neurogenesis-interface-design.md

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
pub struct NeurogenesisOutcome {
    pub births: usize,
    pub deaths: usize,
}

// ── Module principal ────────────────────────────────────────────────────

/// Module de neurogenèse structurelle.
/// Cycle de vie : birth → maturation → pruning protection → homeostatic replacement → synaptic scaling.
pub struct Neurogenesis {
    pub config: NeurogenesisConfig,
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
    /// Le caller ne voit que le résultat, pas les phases.
    pub fn cycle(
        &mut self,
        attractor: &mut AttractorField,
        graph: &mut Graph,
        last_active_step: &[usize],
        noise_std: f64,
    ) -> NeurogenesisOutcome {
        let births = self.birth_phase(attractor, graph, last_active_step, noise_std);
        self.homeostasis(attractor, graph, last_active_step);
        if self.config.synaptic_scaling {
            self.scale_synapses(graph);
        }
        self.end_cycle();
        let deaths = 0; // compté par prune_concepts ailleurs
        NeurogenesisOutcome { births, deaths }
    }

    // ── Phases (Design 2) — privées, mais testables ──

    /// Phase 1 : Naissance de nouveaux concepts
    fn birth_phase(
        &mut self,
        _attractor: &mut AttractorField,
        _graph: &mut Graph,
        _last_active_step: &[usize],
        _noise_std: f64,
    ) -> usize {
        // TODO: migrer depuis sleep_cycle() Phase 1.5
        0
    }

    /// Phase 2 : Homéostasie — remplacer si budget plein
    fn homeostasis(
        &mut self,
        _attractor: &mut AttractorField,
        _graph: &mut Graph,
        _last_active_step: &[usize],
    ) {
        // TODO: migrer depuis sleep_cycle()
    }

    /// Phase 3 : Scaling synaptique
    fn scale_synapses(&self, _graph: &mut Graph) {
        // TODO: migrer depuis sleep_cycle() Phase 3.5
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
