use ndarray::Array1;
use tso_engine::working_memory::WorkingMemory;
use tso_engine::grid_cells::GridCells;
use tso_engine::attractor::AttractorField;
use tso_engine::core::Graph;

fn main() {
    let mut results: Vec<(&str, bool, String)> = Vec::new();
    results.push(test_duallif());
    results.push(test_attractor_classif());
    results.push(test_phi_bounds());
    results.push(test_grid_cells());
    results.push(test_learning());
    println!("
=== SPIKE REPORT ===
");
    for (n, pass, d) in &results {
        println!("{} {}: {}", if *pass { "PASS" } else { "FAIL" }, n, d);
    }
    let p = results.iter().filter(|r| r.1).count();
    println!("
{}/{} pass", p, results.len());
}

fn test_duallif() -> (&'static str, bool, String) {
    let mut wm = WorkingMemory::new(10, 0.95, 0.5);
    let r = wm.observe(&[Array1::zeros(10)]);
    (".Dual-LIF observe returns some", r.is_some(), format!("obs={:?}", r.map(|x| x.0)))
}

fn test_attractor_classif() -> (&'static str, bool, String) {
    let mut af = AttractorField::new(10, 5, 3, 0.1);
    let v = Array1::from_shape_fn(10, |i| (i as f64+1.).sqrt());
    let n = v.dot(&v).sqrt();
    let uv = v.mapv(|x| x / n);
    af.add_prototype(&uv, 0);
    let c = af.predict(&uv);
    (".Attractor predict returns valid class", c < af.n_classes(), format!("class={} n={}", c, af.n_classes()))
}

fn test_phi_bounds() -> (&'static str, bool, String) {
    let mut g = Graph::with_params(0.7, 0.1);
    let id0 = g.add_node(Array1::from_vec(vec![1.0, 0.0]));
    let id1 = g.add_node(Array1::from_vec(vec![0.0, 1.0]));
    g.add_edge(id0, id1, 1);
    let phi = g.phi();
    (".Phi >= 0 with 1 edge", phi >= 0.0, format!("Phi={:.4}", phi))
}

fn test_grid_cells() -> (&'static str, bool, String) {
    let mut gc = GridCells::new(10, 10);
    gc.auto_configure(10, 10);
    (".GridCells extra_dim > 0 for 10x10", gc.extra_dim() > 0, format!("dim={}", gc.extra_dim()))
}

fn test_learning() -> (&'static str, bool, String) {
    let mut af = AttractorField::new(10, 8, 3, 0.01);
    let v = Array1::from_shape_fn(10, |i| (i as f64+1.).sqrt());
    let n = v.dot(&v).sqrt();
    let uv = v.mapv(|x| x/n);
    af.add_prototype(&uv, 0);
    let c = af.predict(&uv);
    let p = af.get_prototype(c);
    let d = p.map(|p| (p - &uv).dot(&(p - &uv)).sqrt()).unwrap_or(1.0);
    (".Attractor learning distance < 1.0", d < 1.0, format!("class={} dist={:.4}", c, d))
}
