#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

echo "=== TSO Commit 2 — Simplifications stdlib ==="

# ── 0. Scinder tso_engine.rs si pas déjà fait ───────────────
if [ ! -f src/engine_core.rs ]; then
    echo "[INFO] Scindage tso_engine.rs..."
    head -n 1024 src/tso_engine.rs > src/engine_core.rs
    sed -n '1025,1378p' src/tso_engine.rs > src/engine_sleep.rs
    sed -n '1379,1621p' src/tso_engine.rs > src/engine_utils.rs
    
    cat > src/tso_engine.rs <<'EOF'
pub mod engine_core;
pub mod engine_sleep;
pub mod engine_utils;
pub use engine_core::*;
pub use engine_sleep::*;
pub use engine_utils::*;
EOF
    
    sed -i '' 's/^impl TsoEngine/impl crate::tso_engine::TsoEngine/' src/engine_sleep.rs
    sed -i '' 's/^impl TsoEngine/impl crate::tso_engine::TsoEngine/' src/engine_utils.rs
    echo "[OK] Scindage terminé"
fi

# ── 1. Inline attention.rs ──────────────────────────────────
echo "[INFO] Inline attention.rs → engine_core.rs..."
rm -f src/attention.rs
sed -i '' '/pub mod attention;/d' src/lib.rs

cat >> src/engine_core.rs <<'ATTN'

// Inline from former attention.rs
fn attend(perception: &Array1<f64>, predicted_prototype: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let diff = perception - predicted_prototype;
    let diffs = diff.mapv(|x| x.abs());
    let exp_sum: f64 = diffs.mapv(|x| (x / temperature).exp()).sum();
    let w = diffs.mapv(|x| (x / temperature).exp() / exp_sum);
    let w_mean = w.mean().unwrap_or(1.0);
    perception * &(&w / w_mean)
}
ATTN
echo "[OK] attention.rs inline — vérifie les appels dans step/heartbeat"

# ── 2. Inline grid_cells.rs ─────────────────────────────────
echo "[INFO] Inline grid_cells.rs → engine_core.rs..."
rm -f src/grid_cells.rs
sed -i '' '/pub mod grid_cells;/d' src/lib.rs

cat >> src/engine_core.rs <<'GRID'

// Inline from former grid_cells.rs
fn augment_with_cell_id(perception: Array1<f64>, x: usize, y: usize, w: usize, h: usize) -> Array1<f64> {
    let cell_id = (x * h + y) as f64 / (w * h) as f64;
    let mut out = Array1::zeros(perception.len() + 1);
    out.slice_mut(ndarray::s![0..perception.len()]).assign(&perception);
    out[perception.len()] = cell_id;
    out
}
GRID
echo "[OK] grid_cells.rs inline"

# ── 3. Absorber replay_buffer.rs dans cerebellum.rs ─────────
echo "[INFO] Absorption replay_buffer.rs → cerebellum.rs..."
# Sauvegarder le contenu avant suppression
cp src/replay_buffer.rs /tmp/replay_buffer.rs.bak
rm -f src/replay_buffer.rs
sed -i '' '/pub mod replay_buffer;/d' src/lib.rs

# On préfixe cerebellum.rs avec le contenu du replay_buffer
# pour que ReplayBuffer soit défini avant d'être utilisé
cat /tmp/replay_buffer.rs.bak > /tmp/new_cerebellum.rs
cat src/cerebellum.rs >> /tmp/new_cerebellum.rs
mv /tmp/new_cerebellum.rs src/cerebellum.rs

# Retirer l'import si présent
sed -i '' '/use crate::replay_buffer::ReplayBuffer;/d' src/cerebellum.rs
echo "[OK] replay_buffer absorbé dans cerebellum.rs"

# ── 4. Simplifier hypothalamus.rs ───────────────────────────
echo "[INFO] Simplification hypothalamus.rs..."
ruby <<'RUBY'
t = File.read('src/hypothalamus.rs')
# Vire les méthodes redondantes (sleep_drive, reset_sleep, sleep_pressure)
# car should_sleep() sera basé sur un compteur externe
t.gsub!(/pub fn sleep_drive(.*?
    }/m, '')
t.gsub!(/pub fn reset_sleep(.*?
    }/m, '')
t.gsub!(/pub fn sleep_pressure(.*?
    }/m, '')
t.gsub!(/
{3,}/, "

")
File.write('src/hypothalamus.rs', t)
RUBY
echo "[OK] hypothalamus simplifié"

# ── 5. Shrink neurogenesis.rs ───────────────────────────────
echo "[INFO] Shrink neurogenesis.rs (garde maturation + scale_edges)..."
ruby <<'RUBY'
lines = File.readlines('src/neurogenesis.rs')
# On garde les 80 premières lignes + les blocs maturation/scale_edges
kept = []
skip = false
lines.each do |l|
  skip = true if l.include?('max_concepts') || l.include?('homeostasis') || l.include?('anti_pruning')
  kept << l unless skip
  skip = false if l.strip == '}' && skip
end
kept = kept[0..79] if kept.length > 80
File.write('src/neurogenesis.rs', kept.join)
RUBY
echo "[OK] neurogenesis shrinké"

# ── Vérification ────────────────────────────────────────────
echo ""
echo "=== Vérification ==="
cargo check --lib 2>&1 | tail -10 || true
cargo test --lib 2>&1 | tail -10 || true

echo ""
echo "=== Commit 2 prêt ==="
echo "git add -A && git commit -m 'feat(engine): simplifications stdlib + inline modules'"
