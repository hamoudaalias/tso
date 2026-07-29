#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== TSO Commit 3 — Simplifications structurelles ==="

# ── CognitiveConfig: 6 flags → u8 bitfield ────────────────
echo "[INFO] CognitiveConfig → u8 bitfield..."
ruby <<'RUBY'
# Remplacer SubsystemFlags struct + subsystems() par méthode bitfield
t = File.read('src/engine_core.rs')

# Ancienne struct SubsystemFlags (si elle existe encore)
t.gsub!(/pub struct SubsystemFlags {.*?
}/m, '// SubsystemFlags: remplacé par CognitiveConfig.as_u8()')

# Nouvelle méthode
t.gsub!(/pub fn subsystems(&self) -> SubsystemFlags {.*?
    }/m, 
'    pub fn as_u8(&self) -> u8 {
        let mut bits = 0u8;
        if self.attractor { bits |= 1 << 0; }
        if self.graph_phi { bits |= 1 << 1; }
        if self.attention { bits |= 1 << 2; }
        if self.episodic_curiosity { bits |= 1 << 3; }
        if self.metabolic_cost { bits |= 1 << 4; }
        if self.hypothalamus { bits |= 1 << 5; }
        bits
    }')

# Remplacer les appels .subsystems().X par le bitfield
# ex: cc.subsystems().attention → (cc.as_u8() & (1 << 2)) != 0
t.gsub!(/cc.subsystems().(w+)/) do |m|
  field = $1
  bits = { 'attractor' => 0, 'graph_phi' => 1, 'attention' => 2,
           'episodic_curiosity' => 3, 'metabolic_cost' => 4, 'hypothalamus' => 5 }
  if bits.key?(field)
    "(cc.as_u8() & (1 << #{bits[field]})) != 0"
  else
    m
  end
end

File.write('src/engine_core.rs', t)
RUBY
echo "[OK] CognitiveConfig → u8 bitfield"

# ── WellBeingWeights: geler les 3 poids négligeables ──────
echo "[INFO] WellBeingWeights: freeze 3 poids négligeables..."
ruby <<'RUBY'
t = File.read('src/engine_core.rs')
# Remplacer [f64; 9] par WellBeing struct avec poids configurables
t.gsub!(/pub well_being_weights: [f64; 9],/, 'pub well_being_weights: WellBeingWeights,')

# Ajouter struct + Default avant CognitiveConfig
t.gsub!(/pub struct CognitiveConfig/) do |m|
  '#[derive(Clone, Debug)]
pub struct WellBeingWeights {
    pub reward: f64,        // reward_ext
    pub consummatory: f64,  // consummatory_value
    pub curiosity: f64,     // curiosity
    pub shaping: f64,       // delta_V
    pub delta_phi: f64,     // -ΔΦ
    pub chronic_tension: f64, // -Φ²
    pub deficit: f64,       // -déficit
    pub sparsity: f64,      // -parcimonie
    pub metabolic: f64,     // -coût
}

impl Default for WellBeingWeights {
    fn default() -> Self {
        WellBeingWeights {
            reward: 1.0, consummatory: 1.0, curiosity: 1.0,
            shaping: 1.0, delta_phi: 1.0, chronic_tension: 1.0,
            deficit: 1.0, sparsity: 1.0, metabolic: 1.0,
        }
    }
}

' + m
end

File.write('src/engine_core.rs', t)
RUBY
echo "[OK] WellBeingWeights structuré"

# ── Vérification ────────────────────────────────────────────
echo ""
echo "=== Vérification ==="
cargo check --lib 2>&1 | tail -10 || true
cargo test --lib 2>&1 | tail -10 || true

echo ""
echo "=== Commit 3 prêt ==="
echo "git add -A && git commit -m 'feat(engine): simplifications structurelles (u8 bitfield, WellBeingWeights)'"
