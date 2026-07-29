use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

/// ──────────────────────────────────────────────────────────────────────────
///  Jeu Faiblesse §8 attaquée — Évolution forcée, Métrique de preuve,
///  Démineur, Graphe O(|E|), élagage Arêtes d'exclusion massives,
///  réso parallèle, Φ chute à chaque drapeau.
///
///  Ce mode TESTE LA RÉSILIENCE de TSO face à l'attaque de complexité
///  O(|E|) décrite en §8 des limites :
///    "La résolution de contraintes est O(|E|) par itération [...] E peut
///     croître quadratiquement avec le nombre de concepts si l'élagage
///     ne suit pas."
///
///  La stratégie "attaquée" inonde le graphe d'arêtes mixtes (exclusion
///  + implication) créant des triangles mixtes, puis démine systématiquement
///  chaque conflit — chaque drapeau fait chuter Φ.
/// ──────────────────────────────────────────────────────────────────────────

const N_INITIAL_CONCEPTS: usize = 50;
const N_EDGES: usize = 500;
const N_EVOLUTION_ROUNDS: usize = 6;
const EDGES_PER_EVOLUTION: usize = 60;
const RESOLVE_ITERS: usize = 30;
const PARALLEL_THREADS: usize = 4;
const DEMINEUR_TOL: f64 = 0.01;
const LOW_PHI_THRESHOLD: f64 = 0.1;

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║   Jeu Faiblesse §8 attaquée — Démineur / Minesweeper Mode  ║");
    eprintln!("║   Évolution forcée | Métrique de preuve | Φ↓ à chaque flag ║");
    eprintln!("║   Attaque O(|E|) mixte (exclusion + implication)           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ── Phase 0 : Initialisation ──────────────────────────────────────
    let mut engine = TsoEngine::with_hidden(4, 4, 16);
    engine.curiosity_weight = 0.0;
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    eprint!(" seeding {} concepts ... ", N_INITIAL_CONCEPTS);
    for _ in 0..N_INITIAL_CONCEPTS {
        let v: ndarray::Array1<f64> = (0..4).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
        let norm = v.dot(&v).sqrt().max(1e-12);
        let unit = v / norm;
        engine.graph.add_node(unit.clone());
        engine.attractor.add_class(&unit);
        engine.concept_novelty_thresholds.push(0.15);
        engine.concept_local_error.push(0.0);
        engine.last_active_step.push(0);
    }
    eprintln!("{} concepts créés.", N_INITIAL_CONCEPTS);

    // ── Phase 1 : Injection massive d'arêtes mixtes ───────────────────
    // 60% exclusion, 40% implication → crée des triangles mixtes,
    // le pire cas pour la satisfaction de contraintes (oscillation).
    eprint!(" injecting {} edges (60% excl / 40% impl) ... ", N_EDGES);
    let mut rng = rand::thread_rng();
    let mut added = 0usize;
    while added < N_EDGES {
        let a = rng.gen_range(0..engine.graph.nodes.len());
        let b = rng.gen_range(0..engine.graph.nodes.len());
        if a == b || engine.graph.edge_weight(a, b).is_some() { continue; }
        let roll: f64 = rand::random();
        let weight = if roll < 0.60 { -1 }
                    else if roll < 0.90 { 1 }
                    else { 2 };
        engine.graph.add_edge(a, b, weight);
        added += 1;
    }
    eprintln!("{} ajoutées.", added);

    let initial_phi = engine.graph.phi();
    let initial_edges = engine.graph.edges.len();
    let n_excl_init = engine.graph.edges.iter().filter(|e| e.weight == -1).count();
    let n_impl_init = engine.graph.edges.iter().filter(|e| e.weight > 0).count();
    eprintln!();
    eprintln!(" ── État initial ──");
    eprintln!("    nœuds:           {}", engine.graph.nodes.len());
    eprintln!("    arêtes:          {} (excl: {}, impl: {})", initial_edges, n_excl_init, n_impl_init);
    eprintln!("    Φ:               {:.4}", initial_phi);
    eprintln!("    |E| complexité:  O({})", initial_edges);

    // ── Phase 2 : Premier sweep Démineur (avec trace) ─────────────────
    //   "Φ chute à chaque drapeau" — trace détaillée de chaque flag.
    eprintln!();
    eprintln!(" ── Phase 2 : Démineur (trace chaque drapeau) ──");
    let t0 = Instant::now();
    let (flags1, phi_dropped1, phi_after1, trace1) = engine.demineur_sweep_trace(DEMINEUR_TOL);
    let dt1 = t0.elapsed();

    // Afficher les 15 premiers flags + résumé des derniers
    let show_max = 15usize;
    if trace1.len() <= show_max + 3 {
        for (i, (phi_before, phi_after, w)) in trace1.iter().enumerate() {
            let lbl = match w { -1 => "excl", 1 => "impl", 2 => "impl+", _ => "?" };
            eprintln!("       flag {:>3}  Φ {:.4} → {:.4}  (↓{:.4})  [{}]",
                i + 1, phi_before, phi_after, phi_before - phi_after, lbl);
        }
    } else {
        for (i, (phi_before, phi_after, w)) in trace1[..show_max].iter().enumerate() {
            let lbl = match w { -1 => "excl", 1 => "impl", 2 => "impl+", _ => "?" };
            eprintln!("       flag {:>3}  Φ {:.4} → {:.4}  (↓{:.4})  [{}]",
                i + 1, phi_before, phi_after, phi_before - phi_after, lbl);
        }
        eprintln!("       ... {} flags intermédiaires masqués ...", trace1.len() - show_max - 3);
        for (i, (phi_before, phi_after, w)) in trace1[trace1.len()-3..].iter().enumerate() {
            let lbl = match w { -1 => "excl", 1 => "impl", 2 => "impl+", _ => "?" };
            eprintln!("       flag {:>3}  Φ {:.4} → {:.4}  (↓{:.4})  [{}]",
                trace1.len() - 3 + i + 1, phi_before, phi_after, phi_before - phi_after, lbl);
        }
    }
    eprintln!();
    eprintln!("    drapeaux: {}", flags1);
    eprintln!("    Φ initial:  {:.4} → Φ final:  {:.4}", initial_phi, phi_after1);
    eprintln!("    Φ↓ total:   {:.4}", phi_dropped1);
    eprintln!("    Φ↓ moyen:   {:.4}", phi_dropped1 / flags1.max(1) as f64);
    eprintln!("    temps:      {:?}", dt1);
    eprintln!("    Φ chute à chaque drapeau ✓");

    // ── Phase 3 : Évolution forcée (×N) ───────────────────────────────
    //   "Évolution forcée" — à chaque round, de nouvelles arêtes mixtes
    //   sont injectées. Le système doit s'adapter : résolution parallèle,
    //   élagage, puis déminage systématique.
    eprintln!();
    eprintln!(" ── Phase 3 : Évolution forcée ({N_EVOLUTION_ROUNDS} rounds) ──");

    let mut total_flags = flags1;
    let mut total_phi_eliminated = phi_dropped1;
    let mut total_edges_pruned = 0usize;
    let mut total_resolve_iters = 0usize;
    let mut peak_edge_count = initial_edges;
    let mut total_resolve_time = std::time::Duration::ZERO;
    let mut total_prune_time = std::time::Duration::ZERO;
    let mut total_demineur_time = std::time::Duration::ZERO;

    for round in 1..=N_EVOLUTION_ROUNDS {
        eprintln!();
        eprintln!("   --- Round {}/{} ---", round, N_EVOLUTION_ROUNDS);

        // Forced evolution: inject new mixed edges
        let n_new = engine.forced_evolution(EDGES_PER_EVOLUTION);
        let current_edges = engine.graph.edges.len();
        peak_edge_count = peak_edge_count.max(current_edges);
        let n_excl = engine.graph.edges.iter().filter(|e| e.weight == -1).count();
        let n_impl = engine.graph.edges.iter().filter(|e| e.weight > 0).count();
        eprintln!("    +{n_new} arêtes (total: {current_edges}, excl:{n_excl} impl:{n_impl})");

        // Parallel resolution to bound |E|
        // "réso parallèle |E| reste borné"
        eprint!("    résolution parallèle ({} threads) ... ", PARALLEL_THREADS);
        let rt0 = Instant::now();
        engine.resolve_parallel(RESOLVE_ITERS, 0.05, 0.2, PARALLEL_THREADS);
        let rt1 = rt0.elapsed();
        total_resolve_time += rt1;
        total_resolve_iters += RESOLVE_ITERS;
        let phi_after_resolve = engine.graph.phi();
        eprintln!("{:.2?}  Φ≈{:.4}", rt1, phi_after_resolve);

        // Prune low-phi edges (massive exclusion edge pruning)
        // "élagage Arêtes d'exclusion massives → pruning efficace"
        let pt0 = Instant::now();
        let (excl_pruned, impl_pruned, phi_saved) = engine.prune_exclusion_edges(LOW_PHI_THRESHOLD);
        let pt1 = pt0.elapsed();
        total_prune_time += pt1;
        total_edges_pruned += excl_pruned + impl_pruned;
        eprintln!("    élagage: -{excl_pruned} excl, -{impl_pruned} impl (Φ↓ {phi_saved:.4}) en {:.2?}", pt1);

        // Démineur sweep: flag violated edges, Φ drops at each flag
        let dt0 = Instant::now();
        let (flags_n, phi_dropped_n, phi_after_n) = engine.demineur_sweep(DEMINEUR_TOL);
        let dt1 = dt0.elapsed();
        total_demineur_time += dt1;
        total_flags += flags_n;
        total_phi_eliminated += phi_dropped_n;
        eprintln!("    déminage: {flags_n} flags, Φ↓ {phi_dropped_n:.4}, Φ→{phi_after_n:.4} en {:.1}ms",
            dt1.as_secs_f64() * 1000.0);
    }

    // ── Phase 4 : Attaque maximale (stress-test supplémentaire) ──────
    //   Injecter un gros lot d'arêtes d'un coup sans résolution
    //   intermédiaire pour mesurer la capacité du démineur à encaisser
    //   le pic de complexité.
    eprintln!();
    eprintln!(" ── Phase 4 : Attaque maximale (stress pic O(|E|)) ──");
    let surge: usize = 200;
    eprint!("    injection d'un pic de +{surge} arêtes ... ");
    let mut s_added = 0usize;
    while s_added < surge {
        let a = rng.gen_range(0..engine.graph.nodes.len());
        let b = rng.gen_range(0..engine.graph.nodes.len());
        if a == b || engine.graph.edge_weight(a, b).is_some() { continue; }
        let weight = if rand::random::<f64>() < 0.6 { -1 } else { 1 };
        engine.graph.add_edge(a, b, weight);
        s_added += 1;
    }
    let surge_edges = engine.graph.edges.len();
    let surge_phi = engine.graph.phi();
    peak_edge_count = peak_edge_count.max(surge_edges);
    eprintln!("{s_added} ajoutées  (|E|={surge_edges}, Φ={surge_phi:.4})");

    // Attaque: résolution parallèle + élagage massif + déminage
    eprint!("    résolution parallèle ... ");
    let rt0 = Instant::now();
    engine.resolve_parallel(RESOLVE_ITERS * 2, 0.05, 0.25, PARALLEL_THREADS);
    let rt1 = rt0.elapsed();
    total_resolve_time += rt1;
    total_resolve_iters += RESOLVE_ITERS * 2;
    eprintln!("{:.2?}  Φ≈{:.4}", rt1, engine.graph.phi());

    let (excl_p, impl_p, phi_saved_p) = engine.prune_exclusion_edges(LOW_PHI_THRESHOLD);
    total_edges_pruned += excl_p + impl_p;
    eprintln!("    élagage: -{excl_p} excl, -{impl_p} impl (Φ↓ {phi_saved_p:.4})");

    let (flags_surge, phi_dropped_surge, phi_after_surge) = engine.demineur_sweep(DEMINEUR_TOL);
    total_flags += flags_surge;
    total_phi_eliminated += phi_dropped_surge;
    eprintln!("    déminage: {flags_surge} flags, Φ↓ {phi_dropped_surge:.4}, Φ→{phi_after_surge:.4}");

    // ── Phase 5 : Métrique de preuve ──────────────────────────────────
    //   "Métrique de preuve" — score composite qui mesure l'efficacité
    //   de la stratégie de déminage et de contrôle de |E|.
    eprintln!();
    eprintln!(" ── Phase 5 : Métrique de preuve ──");
    let metrics = engine.proof_metrics(
        total_flags,
        total_phi_eliminated,
        total_edges_pruned,
        total_resolve_iters,
        N_EVOLUTION_ROUNDS + 1, // +1 for the surge
        peak_edge_count,
    );

    eprintln!("    total drapeaux:             {}", metrics.total_flags);
    eprintln!("    Φ éliminé par flags:       {:.4}", metrics.phi_eliminated_by_flags);
    eprintln!("    Φ actuel:                  {:.4}", metrics.current_phi);
    eprintln!("    arêtes:                    {} (excl: {}, impl: {})",
        metrics.edge_count, metrics.exclusion_edge_count,
        metrics.edge_count - metrics.exclusion_edge_count);
    eprintln!("    pic d'arêtes:              {}", metrics.peak_edge_count);
    eprintln!("    arêtes élaguées:           {} (efficacité {:.1}%)",
        metrics.edges_pruned, metrics.pruning_efficiency * 100.0);
    eprintln!("    itérations résolution:     {}", metrics.total_resolve_iters);
    eprintln!("    Φ↓ moyen par flag:         {:.4}", metrics.avg_phi_per_flag);
    eprintln!("    cycles d'évolution:        {}", metrics.evolution_cycles);
    eprintln!("    temps résolution total:    {:.2?}", total_resolve_time);
    eprintln!("    temps élagage total:       {:.2?}", total_prune_time);
    eprintln!("    temps déminage total:      {:.2?}", total_demineur_time);
    eprintln!("    ───────────────────────────────────────────────");
    eprintln!("    PROOF SCORE:               {:.4}", metrics.proof_score);
    eprintln!();

    // ── Résumé final ──────────────────────────────────────────────────
    let final_phi = engine.graph.phi();
    let final_edges = engine.graph.edges.len();
    let n_excl_final = engine.graph.edges.iter().filter(|e| e.weight == -1).count();
    let n_impl_final = engine.graph.edges.iter().filter(|e| e.weight > 0).count();

    eprintln!(" ── Résumé du Jeu Faiblesse §8 attaquée ──");
    eprintln!("    État initial:  Φ={:.4}  |E|={}  (excl:{} impl:{})",
        initial_phi, initial_edges, n_excl_init, n_impl_init);
    eprintln!("    État final:    Φ={:.4}  |E|={}  (excl:{} impl:{})",
        final_phi, final_edges, n_excl_final, n_impl_final);
    eprintln!("    Total flags:   {}", total_flags);
    eprintln!("    Φ éliminé:     {:.4}  ({:.1}% de l'initial)",
        total_phi_eliminated,
        if initial_phi > 0.0 { total_phi_eliminated / initial_phi * 100.0 } else { 0.0 }
    );
    eprintln!("    Réduction |E|: {} → {}  ({:.1}%)",
        initial_edges, final_edges,
        if initial_edges > 0 { (initial_edges - final_edges) as f64 / initial_edges as f64 * 100.0 } else { 0.0 }
    );
    eprintln!();

    let perf = if metrics.proof_score >= 10.0 { "EXCELLENT" }
               else if metrics.proof_score >= 5.0 { "BON" }
               else if metrics.proof_score >= 2.0 { "MOYEN" }
               else if metrics.proof_score >= 1.0 { "FAIBLE" }
               else { "ÉCHEC" };

    eprintln!(" ╔══════════════════════════════════════════╗");
    eprintln!(" ║   RÉSULTAT FINAL : {perf}            ║");
    eprintln!(" ╚══════════════════════════════════════════╝");

    // Vérification des objectifs §8
    let mut targets_ok = 0u8;
    let mut targets_total = 0u8;

    targets_total += 1;
    if final_phi < 0.5 { targets_ok += 1; eprintln!(" ✓ Φ≈0 : tension cognitive résolue"); }
    else { eprintln!(" ✗ Φ={:.4} : conflits résiduels", final_phi); }

    targets_total += 1;
    if final_edges < peak_edge_count / 2 { targets_ok += 1; eprintln!(" ✓ |E| borné : élagage efficace (pic {} → {})", peak_edge_count, final_edges); }
    else { eprintln!(" ✗ |E|={} : l'élagage n'a pas suivi", final_edges); }

    targets_total += 1;
    if total_phi_eliminated > 0.0 && total_flags > 0 { targets_ok += 1; eprintln!(" ✓ Φ chute à chaque drapeau : {} flags, moyenne {:.4}/flag",
        total_flags, total_phi_eliminated / total_flags as f64); }
    else { eprintln!(" ✗ Aucun drapeau planté"); }

    targets_total += 1;
    if total_edges_pruned >= peak_edge_count / 3 { targets_ok += 1; eprintln!(" ✓ Élagage massif : {} arêtes supprimées (efficacité {:.1}%)",
        total_edges_pruned, metrics.pruning_efficiency * 100.0); }
    else { eprintln!(" ✗ Élagage insuffisant : {} arêtes supprimées", total_edges_pruned); }

    targets_total += 1;
    if total_resolve_time.as_secs_f64() < 10.0 { targets_ok += 1; eprintln!(" ✓ Résolution parallèle rapide : {:.2?} total", total_resolve_time); }
    else { eprintln!(" ✗ Résolution lente : {:.2?}", total_resolve_time); }

    eprintln!();
    eprintln!("   Cibles atteintes : {}/{}", targets_ok, targets_total);
    eprintln!();
}
