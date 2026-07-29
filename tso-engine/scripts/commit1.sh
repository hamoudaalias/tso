#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT/tso-engine"

echo "=== TSO Commit 1 — Nettoyage code mort ==="

# ── 1. Suppressions ──
echo "[1/5] Suppression fichiers morts..."
rm -f src/constraint_redirection.rs
rm -f src/multi_grid_cells.rs
rm -f src/bin/vae_trainer.rs.bak
echo "  ✓ done"

# ── 2. Archive binaires ──
echo "[2/5] Archivage binaires expérimentaux..."
mkdir -p specs/archive/bin
KEEP="debug_rl weakness_game_v3 eval_minigrid"
for f in src/bin/*.rs; do
    basename=$(basename "$f" .rs)
    if ! echo "$KEEP" | grep -qw "$basename"; then
        mv "$f" specs/archive/bin/
    fi
done
echo "  ✓ conservés: $KEEP"

# ── 3. lib.rs ──
echo "[3/5] lib.rs..."
LIB="src/lib.rs"
sed -i '' '/pub mod multi_grid_cells;/d' "$LIB"
sed -i '' '/pub mod constraint_redirection;/d' "$LIB"
ruby -i -pe 'gsub(/^pub mod fpi;/, "#[cfg(feature = "active-inference")]\npub mod fpi;")' "$LIB"
ruby -i -pe 'gsub(/^pub mod efe;/, "#[cfg(feature = "active-inference")]\npub mod efe;")' "$LIB"
ruby -i -pe 'gsub(/^pub mod inference;/, "#[cfg(feature = "active-inference")]\npub mod inference;")' "$LIB"
ruby -i -pe 'gsub(/^pub mod vae;/, "#[cfg(feature = "vae-encoder")]\npub mod vae;")' "$LIB"
echo "  ✓ done"

# ── 4. Cargo.toml ──
echo "[4/5] Cargo.toml..."
cat > Cargo.toml << 'TOML'
[package]
name = "tso-engine"
version = "0.1.0"
edition = "2024"

[dependencies]
ndarray = { version = "0.16", features = ["serde"] }
rand = "0.8"
serde = { version = "1", features = ["derive"] }
bincode = "1.3"
ctrlc = "3.4"
serde_json = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }

[dev-dependencies]
pyo3 = { version = "0.29", features = ["auto-initialize"] }

[features]
default = ["cognitive-cycle"]
cognitive-cycle = []
active-inference = []
vae-encoder = []
parallel-resolve = []
experimental-bins = []
interop = []
TOML
echo "  ✓ done"

# ── 5. Vérification ──
echo "[5/5] Vérification..."
cargo check --lib 2>&1 | tail -3
cargo check --bins 2>&1 | tail -3
cargo test 2>&1 | tail -3

echo ""
echo "=== Commit 1 prêt ==="
echo "git add -A && git commit -m 'feat(engine): mort code + feature gates'"
