---
name: graph-world-bevy
description: 'Rust + Bevy (headless app/ecs/time only) implementation workflow for GraphOS world projects. Use after graph-world closes graph design, and use graph-management for graph reads/mutations before bootstrapping GraphOS project files and initializing a Rust runtime without Bevy rendering stack.'
argument-hint: 'Describe your Bevy world task, e.g. "bootstrap GraphOS + Rust bevy_app/bevy_ecs/bevy_time project in current directory and create World bootstrap systems"'
user-invocable: true
---

# GraphOS Skill: graph-world-bevy

Implement GraphOS world runtime in Rust with Bevy ECS/time only, without Bevy rendering, windowing, UI, or asset pipeline.

This skill depends on `graph-world` for graph modeling and closure validation, and on `graph-management` for graph inspection, validation, and atomic graph mutations. Use `graph-world` first whenever the request changes `World`/`Context`/`Variant`/`System`/`Event`/`EventSystem` topology, then use `graph-management` to read or apply the resulting graph changes.

## Scope Boundary

- This skill covers GraphOS project bootstrap plus Rust runtime initialization.
- This skill does not include TypeScript implementation code.
- This skill does not include Bevy rendering stack (`bevy_render`, `bevy_winit`, `bevy_pbr`, `bevy_ui`, `bevy_sprite`, etc.).
- If the request includes rendering or frontend integration, hand off to another presentation/runtime skill.

## When to Use

- You need a GraphOS world project with Rust runtime instead of TypeScript runtime.
- You need Bevy as ECS/runtime scheduler and time source only.
- You need a deterministic, headless app loop for world/domain logic.

## Preconditions

1. Complete graph design with `graph-world` first.
2. Finish graph closed-loop validation before runtime coding.
3. Use `graph-management` to inspect graph state, validate node types/plugins, and apply any graph mutations before changing Rust runtime bindings.
4. If Graph changed, regenerate/update GraphOS outputs before changing Rust runtime bindings.

## Project Bootstrap (GraphOS, No TypeScript Code)

Use this when initializing a new world project that still needs GraphOS CLI and graph metadata, but no TS runtime code.

1. Initialize npm metadata and GraphOS tooling:

```bash
npm init -y
npm i -D graphos-world-plugin graphos-cli
```

2. Ensure `package.json` contains GraphOS config with TS generators disabled:

```json
{
  "type": "module",
  "graphos": {
    "world": {
      "genTypeScript": {
        "enabled": false,
        "outDir": "gen"
      },
      "genWebTypeScript": {
        "enabled": false,
        "outDir": "app"
      },
      "genCocosCreator": {
        "enabled": false,
        "outDir": "../cocos/assets/gen"
      },
      "genBevy": {
        "enabled": true,
        "outDir": "src/gen"
      }
    }
  }
}
```

3. Ensure `package.json` scripts includes GraphOS, wasm, and cross-compile commands:

```json
{
  "scripts": {
    "graphos": "graphos",
    "build:wasm": "wasm-pack build . --target web --release --out-dir dist",
    "build:wasm:bundler": "wasm-pack build . --target bundler --release --out-dir dist",
    "build:wasm:node": "wasm-pack build . --target nodejs --release --out-dir dist",

    "build:ios:arm64": "cargo build --release --target aarch64-apple-ios",
    "build:ios:x86_64": "cargo build --release --target x86_64-apple-ios",
    "build:ios:arm64_sim": "cargo build --release --target aarch64-apple-ios-sim",

    "build:android:armv7": "cargo ndk -t armeabi-v7a build --release",
    "build:android:arm64": "cargo ndk -t arm64-v8a build --release",

    "build:macos:x86_64": "cargo build --release --target x86_64-apple-darwin",
    "build:macos:arm64": "cargo build --release --target aarch64-apple-darwin",

    "build:windows:x86_64": "cargo zigbuild --release --target x86_64-pc-windows-gnu",
    "build:linux:x86_64": "cargo zigbuild --release --target x86_64-unknown-linux-gnu"
  }
}
```

4. Create `World.graph.json` in project root:

```json
{
  "id": "main",
  "name": "World",
  "nodes": [],
  "edges": []
}
```

5. Optional verification:

```bash
npm run graphos -- --help
npm run build:wasm:node
npm run build:macos:arm64
```

Completion checks:
- GraphOS CLI is available.
- Graph file exists.
- No TypeScript build/runtime files are required by this skill.
- wasm-pack build scripts are available in `package.json`.
- Cross-compile scripts exist for iOS/Android/macOS/Windows/Linux targets.

## Rust + Bevy Initialization (Latest, Headless)

Goal: initialize a Rust runtime that uses Bevy ECS/time scheduling only.

### Step 1: Initialize Rust crate in current directory

```bash
# Run this in the project root directory.
cargo init --bin .
```

If a Cargo project already exists in the current directory, skip this step.

### Cargo.toml Rules (Mandatory)

Apply these rules in `Cargo.toml` for this skill:

1. `package.name` MUST be fixed to `"app"`.
2. Library output MUST support both crate types: `"cdylib"` and `"rlib"` and `"staticlib"`.

Example:

```toml
[package]
name = "app"

[lib]
crate-type = ["cdylib", "rlib", "staticlib"]
```

### Step 2: Add world-bevy and Bevy runtime dependencies

```bash
cargo add world-bevy
cargo add bevy --no-default-features --features bevy_app,bevy_ecs,bevy_time
```

Fallback (if feature flags change in newer Bevy releases):

```bash
cargo add world-bevy
cargo add bevy_app bevy_ecs bevy_time
```

Rules:
- Always use the latest published versions.
- Do not enable rendering/window/UI-related features.
- Keep dependency set minimal: app, ecs, time.
- `world-bevy` is a mandatory dependency; do not skip it.
- See [world-bevy Dependency](#world-bevy-dependency) below for what the crate provides.

### world-bevy Dependency

```bash
cargo add world-bevy
```

#### What world-bevy provides

| Item | Path | Role |
|---|---|---|
| `Context` | `world_bevy::core::Context` | Core entity identity component (`id`, `table`, `pid`) |
| `IVariant` | `world_bevy::core::IVariant` | Trait for graph variant components; implement this on generated variant types |
| `IEvent` | `world_bevy::core::IEvent` | Trait for graph event types; implement this on generated event types |
| `Message` | `world_bevy::core::Message` | Inbound/outbound message enum (`Spawn`, `Despawn`, `Change`, `Event`, `Reset`) |
| `WorldResource` | `world_bevy::core::WorldResource` | Global runtime resource managing entity registry, variant tables, event dispatch, and message queues |
| `next_id()` | `world_bevy::core::next_id` | Generate unique entity IDs |
| `reg()` | `world_bevy::core::reg` | Register core world systems (`PreUpdate` message pump, `PostUpdate` spawn/change/despawn ordering) |
| `reg_veriant()` | `world_bevy::core::reg_veriant` | Register a variant table entry for CBOR-driven component changes |
| `reg_event()` | `world_bevy::core::reg_event` | Register an event type for CBOR-driven event dispatch |
| `ffi` | `world_bevy::ffi` | C FFI bindings for wasm host integration (`ffi_app_create`, `ffi_app_update`, `ffi_app_inbound`, `ffi_app_outbound`, `ffi_app_exit`) |

The project's `src/core/mod.rs` re-exports world-bevy types and adds project-specific helpers:

```rust
// src/core/mod.rs
pub use world_bevy::core::*;

use bevy_ecs::prelude::*;

#[derive(Event)]
pub struct WorldLogEvent {
    pub log: String,
}
```

This ensures that all `world_bevy::core::*` imports (used throughout `src/gen` and `src/app`) resolve to `world_bevy::core` types plus any project-local extensions.

**Dependency rules:**
- `world-bevy` MUST be listed in `Cargo.toml` before generating or compiling `src/gen`.
- Generated `src/gen/world.rs` code uses `world_bevy::core::Context`, `world_bevy::core::IVariant`, and `world_bevy::core::IEvent` — these resolve through the `src/core/mod.rs` re-export chain.
- Do not manually re-implement `Context`, `Message`, `WorldResource`, or the variant/event registration infrastructure — they are provided by `world-bevy`.
- The `world-bevy` crate already depends on `bevy_app`, `bevy_ecs`, and `bevy_time`; the project crate only needs a direct `bevy` dependency when using additional Bevy plugins not covered by world-bevy's transitive dependencies.

### Step 3: Minimal runtime app (no renderer)

Before wiring systems, initialize base source files for generated modules.

`src/gen/mod.rs`:

```rust
pub mod world;
```

`src/lib.rs`:

```rust
pub mod r#gen;
pub mod app;
```

Then implement runtime entry logic.

`src/main.rs` example:

```rust
use bevy_app::{App, Startup, Update};
use bevy_ecs::prelude::*;
use bevy_time::{Time, TimePlugin};

#[derive(Component, Default)]
struct TickAge(f32);

fn setup(mut commands: Commands) {
    commands.spawn(TickAge::default());
}

fn tick(time: Res<Time>, mut q: Query<&mut TickAge>) {
    for mut age in &mut q {
        age.0 += time.delta_secs();
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins(TimePlugin);
    world_bevy::core::reg(&mut app);
    app.add_systems(Startup, setup);
    app.add_systems(Update, tick);
    app.run();
}
```

**Important:** `world_bevy::core::reg(&mut app)` must be called before any other system registration that depends on `WorldResource`. It inserts the `WorldResource` and sets up the message pump (`PreUpdate`) and spawn/change/despawn ordering (`PostUpdate`).

### Step 4: Build verification

```bash
cargo check
cargo run
npm run build:wasm:node
```

## Workflow

### File Organization Rules (Mandatory)

Every graph `System` and `EventSystem` MUST follow these rules with no exceptions:

1. **One file per System/EventSystem** — each System or EventSystem lives in its own `.rs` file. Never put multiple Systems/EventSystems in the same file.
2. **File naming: lowercase + underscores (snake_case)** — match the graph node name converted to snake_case. Example: graph node `StartGameEventSystem` → file `start_game_event_system.rs`.
3. **Directory: `src/app/`** — all System and EventSystem implementation files go under `src/app/`. Do not place them in `src/gen/`, `src/core/`, or any other directory.
4. **Registration: `src/app/mod.rs`** — every System/EventSystem module must be declared (`pub mod ...`) and its `reg(app)` called inside `src/app/mod.rs`. No implicit or auto-registration.
5. **World startup spawn hook (mandatory): `src/app/mod.rs`** — `reg(app)` MUST include `app.add_systems(Startup, on_world_spawn);`, and `src/app/mod.rs` MUST define `fn on_world_spawn(mut commands: Commands)` that spawns `WorldContext`.
6. **Types & enums: `src/app/types.rs`** — define all shared types, enums, and constants here. This includes: singleton entity IDs (e.g. `const GAME_ID: &str = "game"`), state enums (e.g. `enum GameState { ... }`), event type enums, and any constant strings used across multiple Systems/EventSystems. Do not scatter these definitions across individual System files.
7. **Config definitions: `src/app/config.rs`** — define configuration structs here. Every tunable logic parameter (speeds, durations, thresholds, sizes, probabilities) must live in a config struct, never as a hardcoded magic number in System logic. Example: `struct GameConfig { pub move_speed: f32, pub jump_force: f32 }`.
8. **Shared helper functions: `src/app/common.rs`** — all reusable helper functions shared by multiple Systems/EventSystems MUST be implemented in `src/app/common.rs`. Do not duplicate common logic across System files.
9. **Config instances: `src/config/*.rs` + `src/config/mod.rs`** — concrete config values go under `src/config/` as separate files, split by domain (e.g. one file per map/scene: `src/config/map_city.rs`, `src/config/map_dungeon.rs`). `src/config/mod.rs` aggregates all config modules (`pub mod map_city; pub mod map_dungeon; ...`) and optionally re-exports them. Systems read config via Bevy `Resource`, never by importing config files directly — this keeps Systems testable with different configs.

Directory layout example:

```text
src/
  app/
    mod.rs                       # declares all modules + calls reg(app) for each
    types.rs                     # shared types, enums, constants (rule 6)
    config.rs                    # config struct definitions (rule 7)
    common.rs                    # shared helper functions (rule 8)
    world_bootstrap_system.rs    # System: WorldBootstrapSystem
    game_system.rs               # System: GameSystem
    scene_system.rs              # System: SceneSystem (if needed)
    start_game_event_system.rs   # EventSystem: StartGameEventSystem
  config/
    mod.rs                    # aggregates all config modules
    game_config.rs            # global gameplay params
    map_city.rs               # city map config
    map_dungeon.rs            # dungeon map config
  gen/
    mod.rs
    world.rs                     # generated — never edit
  core/
    mod.rs                     # re-exports world_bevy::core::* + project-local helpers (WorldLogEvent, etc.)
  lib.rs
  main.rs
```

When adding a new System/EventSystem:
1. Create `src/app/<snake_case_name>.rs`
2. Add `pub mod <snake_case_name>;` to `src/app/mod.rs`
3. Add `<snake_case_name>::reg(app);` inside the `reg()` function in `src/app/mod.rs`

**Ordering rule:** Before implementing any System or EventSystem logic, `src/app/types.rs`, `src/app/config.rs`, and `src/app/common.rs` must be fully defined first. Systems consume shared types, config, and helper functions; never write System logic against missing shared modules. The correct workflow order is: **types → config → common → systems**.

### Step 1: Sync Graph Output Into Rust

1. Confirm Graph changes are complete and validated in `graph-world`.
2. Regenerate `src/gen` if Graph changed.
3. Re-read generated `src/gen/*.rs` before implementing runtime logic.

Completion checks:
- Generated `Context`, `Variant`, and `Event` types match the current graph.
- No `src/app` code is written against stale generated APIs.

### Step 2: Implement Systems In `src/app/`

Implement concrete lifecycle `System` behavior under `src/app/` after Graph ownership is finalized.

**Prerequisite:** `src/app/types.rs`, `src/app/config.rs`, and `src/app/common.rs` must be fully defined before writing any System logic (see [Ordering rule](#file-organization-rules-mandatory)).

**Follow the [File Organization Rules](#file-organization-rules-mandatory) strictly.** Each graph System gets its own file in `src/app/`, named in snake_case, and registered in `src/app/mod.rs`.

Reference implementation for this skill lives in:
- `skills/graph-world-bevy/src/app/`
- structure aligned with `/Volumes/SSD_1T/src/abpilot-cc/RPG-Platformer/src/app`

Lifecycle systems in Bevy should be wired with observers and ECS queries, not by editing `src/gen`.

```rust
use world_bevy::core::Context;
use crate::r#gen::world::*;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_time::{Time, Virtual};

pub fn reg(app: &mut App) {
    app.add_systems(Update, on_update_system);
    app.add_observer(on_spawn_system);
    app.add_observer(on_despawn_system);
}

fn on_update_system(
    mut query: Query<(&Context, &mut GameTimeComponent), With<GameContext>>,
    time: Res<Time<Virtual>>,
) {
    for (_context, mut game_time) in &mut query {
        game_time.0 += time.delta_secs_f64();
    }
}

fn on_spawn_system(query: Query<(Entity, &Context), Added<GameContext>>) {
    for (_entity, _context) in &query {
        // TODO system-specific spawn logic
    }
}

fn on_despawn_system(removed: RemovedComponents<GameContext>) {
    for _entity in removed.read() {
        // TODO system-specific cleanup logic
    }
}
```

Implementation guidance:
- Put each graph `System` into its own Rust module under `src/app/`.
- Use snake_case file names for Rust modules and keep them aligned with graph names.
- Example: graph node `GameSystem` -> `src/app/game_system.rs`.
- Register per-system observers from `pub fn reg(app: &mut App)`.
- Use `Added<YourContext>` for spawn semantics.
- Use `RemovedComponents<YourContext>` for despawn semantics.
- **Prohibited: never use `EventReader<T>` or `EventWriter<T>`.** Handle events exclusively with `On<T>` triggers + `app.add_observer(...)`.
- Use `Res<Time<Virtual>>` for deterministic update timing.
- For simulator-visible runtime logs, emit `world_bevy::core::WorldLogEvent` via `Commands`:

```rust
commands.trigger(world_bevy::core::WorldLogEvent {
  log: "World context added".to_string(),
});
```

- This event-based logging is the canonical way to produce records visible from `GET /api/world/log`; do not rely on `println!` for simulator log verification.
- Query generated variant components directly; do not mirror generated state into ad-hoc global caches.
- Keep temporary working data local to the observer function or a dedicated Bevy `Resource` owned by runtime code.
- Do not modify generated files under `src/gen`.

Completion checks:
- Implemented `System` modules compile against the current generated Rust types.
- `System` logic remains in `src/app/` and does not leak into `src/gen/`.
- After `System` changes, run `npm run build:wasm:node`.

### Step 3: Implement World Startup Entry

Goal: define a deterministic startup entry that bootstraps required root/domain contexts.

You must implement startup initialization in a dedicated bootstrap module under `src/app/`.

```rust
use world_bevy::core::Context;
use crate::r#gen::world::*;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

const GAME_ID: &str = "game";
const SCENE_ID: &str = "scene";

pub fn reg(app: &mut App) {
    app.add_observer(on_world_spawn_system);
}

fn on_world_spawn_system(
    query: Query<&Context, (With<WorldContext>, Added<WorldContext>)>,
    existing_game: Query<&Context, With<GameContext>>,
    existing_scene: Query<&Context, With<SceneContext>>,
    mut commands: Commands,
) {
    for world in &query {
        let has_game = existing_game.iter().any(|ctx| ctx.id == GAME_ID);
        if !has_game {
            GameContext::spawn(
                &mut commands,
                Some(world.id.clone()),
                GAME_ID,
                GameTimeComponent(0.0),
                GameStateComponent(0),
            );
        }

        let has_scene = existing_scene.iter().any(|ctx| ctx.id == SCENE_ID);
        if !has_scene {
            SceneContext::spawn(
                &mut commands,
                Some(world.id.clone()),
                SCENE_ID,
                SceneGridComponent(SceneGridSchema {
                    width: 0,
                    height: 0,
                    tile_size: 1.0,
                    tiles: Vec::new(),
                }),
            );
        }
    }
}
```

Startup guidance:
- Create a bootstrap `System` under `World` in graph design.
- Register its module in `src/app/mod.rs`.
- Use generated `Context::spawn(...)` helpers from `src/gen`.
- Keep initialization idempotent by checking existing singleton ids before spawning.
- Prefer fixed ids for singleton/root contexts.
- Use `pid` to attach child contexts to the owning parent context id.
- Do not resolve singleton contexts by positional child order.

### Step 4: Implement EventSystems In `src/app/`

Implement graph `EventSystem` handlers as Bevy observers over generated events.

**Prerequisite:** `src/app/types.rs`, `src/app/config.rs`, and `src/app/common.rs` must be fully defined first (see [Ordering rule](#file-organization-rules-mandatory)).

**Follow the [File Organization Rules](#file-organization-rules-mandatory) strictly.** Each EventSystem gets its own file in `src/app/`, named in snake_case (e.g. `StartGameEventSystem` → `start_game_event_system.rs`), and registered in `src/app/mod.rs`.

```rust
use world_bevy::core::Context;
use crate::r#gen::world::*;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub fn reg(app: &mut App) {
    app.add_observer(on_start_game_event_system);
}

fn on_start_game_event_system(
    trigger: On<StartGameEvent>,
    mut query: Query<(&Context, &mut GameStateComponent), With<GameContext>>,
) {
    let event = trigger.event();

    for (_context, mut game_state) in &mut query {
        game_state.0 = match event.game_type.as_str() {
            "running" => 1,
            "paused" => 2,
            _ => 0,
        };
    }
}
```

Implementation guidance:
- Put each graph `EventSystem` into its own Rust module under `src/app/`.
- Example: graph node `StartGameEventSystem` -> `src/app/start_game_system.rs`.
- **Prohibited: NEVER use `EventReader<T>` or `EventWriter<T>` to read/write events.** Use `On<T>` with `app.add_observer(...)` as the only event handling mechanism.
- Register event handlers with `app.add_observer(...)`.
- Use `On<GeneratedEvent>` as the trigger type.
- Read payload from `trigger.event()`.
- If the handler needs to expose runtime behavior to simulator logs, call `commands.trigger(world_bevy::core::WorldLogEvent { ... })` in the handler path you want to verify.
- Fetch target contexts/components via ECS `Query`; mutate only the scope owned by the graph.
- If event payload schema changes in Graph, regenerate `src/gen` first, then update handler signatures.
- Do not invent new topology in runtime code to compensate for missing graph nodes; go back to `graph-world`.

Completion checks:
- Implemented `EventSystem` modules compile against the current generated event payloads.
- Event handlers only mutate graph-owned scope and stay under `src/app/`.
- After `EventSystem` changes, run `npm run build:wasm`.

### Step 5: Register Runtime Wiring In `src/app/mod.rs`

After implementing handlers, wire the runtime explicitly in `src/app/mod.rs`.

```rust
use bevy_app::prelude::*;

pub mod config;
pub mod game_system;
pub mod start_game_system;
pub mod types;
pub mod world_bootstrap_system;

pub fn reg(app: &mut App) {
    world_bootstrap_system::reg(app);
    game_system::reg(app);
    start_game_system::reg(app);
}
```

Registration guidance:
- Declare `pub mod types;`, `pub mod config;`, and `pub mod common;` in `src/app/mod.rs`.
- Register every implemented `System` module in `src/app/mod.rs`.
- Register every implemented `EventSystem` module in `src/app/mod.rs`.
- Insert concrete config instances (from `src/config/`) as Bevy `Resource` in `main.rs`, not inside `src/app/mod.rs`.
- Keep `src/lib.rs` or `src/main.rs` responsible only for high-level runtime assembly:

```rust
use bevy_app::prelude::*;
use bevy_time::TimePlugin;

mod config;

fn main() {
    let mut app = App::new();
    app.add_plugins(TimePlugin);
    // Register world-bevy core infrastructure (WorldResource, message pump, spawn/change/despawn systems)
    world_bevy::core::reg(&mut app);
    // Insert concrete config instances as Resources (rule 7)
    app.insert_resource(config::game_config::GAME_CONFIG);
    world_bevy::core::reg(&mut app);
    crate::r#gen::world::reg(&mut app);
    crate::app::reg(&mut app);
    app.run();
}
```

Completion checks:
- Every implemented `System` and `EventSystem` module is registered.
- `src/app/mod.rs` matches current graph contracts.
- Runtime assembly does not require edits inside `src/gen`.
- After runtime wiring changes, run `npm run build:wasm`.

### Step 5: wasm-pack packaging

1. Install wasm target and wasm-pack:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

2. Ensure root `Cargo.toml` exposes cdylib:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

3. Build/package with npm scripts:

```bash
npm run build:wasm
# or
npm run build:wasm:bundler
npm run build:wasm:node
```

4. Optional direct command:

```bash
wasm-pack build . --target web --release --out-dir pkg
```

Completion checks:
- Runtime compiles and runs.
- ECS systems execute.
- Time resource updates normally.
- No Bevy render/window/UI modules are linked.
- `pkg/` contains wasm-pack artifacts (`.wasm`, JS glue, package metadata).

### Step 6: Cross-compile setup on macOS

1. Install Rust targets:

```bash
rustup target add \
  aarch64-apple-ios \
  x86_64-apple-ios \
  aarch64-apple-ios-sim \
  armv7-linux-androideabi \
  aarch64-linux-android \
  x86_64-apple-darwin \
  aarch64-apple-darwin \
  x86_64-pc-windows-gnu \
  x86_64-unknown-linux-gnu
```

2. Install cross-compile helpers:

```bash
brew install zig
cargo install cargo-zigbuild
cargo install cargo-ndk
```

3. Configure Android NDK (required by `cargo ndk`):

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
export ANDROID_NDK_ROOT="$ANDROID_NDK_HOME"
```

4. Run platform builds:

```bash
# iOS
npm run build:ios:arm64
npm run build:ios:x86_64
npm run build:ios:arm64_sim

# Android
npm run build:android:armv7
npm run build:android:arm64

# macOS
npm run build:macos:x86_64
npm run build:macos:arm64

# Windows / Linux (cross on macOS via zig)
npm run build:windows:x86_64
npm run build:linux:x86_64
```

Output notes:
- Rust artifacts are generated under `target/<triple>/release/`.
- Android `cargo ndk` outputs ABI-specific artifacts for `armeabi-v7a` and `arm64-v8a`.
- For iOS universal libs, combine device/simulator outputs later with Xcode tooling as needed.

## GraphOS Integration Notes for Rust Runtime

- Keep graph topology/design operations in `graph-world` workflow.
- Keep runtime implementation in Rust under this skill.
- If Graph model changes, apply Graph changes first, then update Rust side adapters/systems.
- Do not introduce TypeScript runtime code in this workflow.
- Implement manual runtime logic under `src/app/`; keep generated code isolated under `src/gen/`.
- Use the reference modules in `skills/graph-world-bevy/src/app/` as the baseline pattern for System/EventSystem wiring.

## Quality Gates

1. Graph-first gate: runtime coding starts only after graph closure validation passes.
2. No-TS gate: no TypeScript runtime implementation is introduced.
3. Headless-Bevy gate: no rendering/window/UI Bevy modules enabled; `world-bevy` dependency is present in `Cargo.toml`.
4. Minimal-runtime gate: app, ecs, time capabilities are present and verified; `world_bevy::core::reg(&mut app)` is called in `main.rs`.
5. Generated-code gate: `src/gen` has been regenerated and re-read after Graph changes.
6. App-wiring gate: all manual runtime logic lives in `src/app/`, not `src/gen/`; each System/EventSystem is in its own file with snake_case naming.
7. Registration gate: every System/EventSystem module is wired through `src/app/mod.rs`.
8. Event-observer gate: `EventReader<T>` and `EventWriter<T>` are prohibited. All event handling MUST use `On<T>` triggers registered via `app.add_observer(...)`. No `Events<T>` resource reads/writes anywhere in the codebase.
9. Types gate: shared types, enums, constants, and singleton IDs are centralized in `src/app/types.rs`; not scattered across System files.
10. Common gate: reusable helper logic is centralized in `src/app/common.rs`; duplicated helper implementations across System/EventSystem files are prohibited.
11. Config gate: all tunable logic parameters live in config structs (`src/app/config.rs`); no magic numbers in System logic; concrete config instances exist under `src/config/`.
12. Startup gate: world bootstrap logic is idempotent and singleton-safe.
13. Build gate: `cargo check` succeeds for the Rust runtime crate.
14. wasm wiring gate: after any `System` / `EventSystem` / `src/app/mod.rs` change, `npm run build:wasm` succeeds.
15. wasm gate: `wasm-pack build` succeeds and outputs package files.
16. cross-compile gate: all required target scripts complete successfully on macOS toolchain.

## Failure Recovery

- `world_bevy::core::Context` or `world_bevy::core::IVariant` not found:
  ensure `src/core/mod.rs` contains `pub use world_bevy::core::*;` and `Cargo.toml` lists `world-bevy` as a dependency.
- `world_bevy::core::reg` not called:
  `WorldResource` will be missing, causing spawn/change/despawn systems to panic. Always call `world_bevy::core::reg(&mut app)` in `main.rs` before any system that queries `Res<WorldResource>`.
- `cargo add bevy --no-default-features ...` fails due to feature changes:
  use split crates `bevy_app`, `bevy_ecs`, `bevy_time` instead.
- Generated Rust types do not match expected Context/Event names:
  regenerate `src/gen` first, then reopen the generated files before editing `src/app`.
- Runtime logic was added into generated files:
  move that code into `src/app/` modules and keep `src/gen` generated-only.
- `System` or `EventSystem` edits compile in native Rust but fail in wasm packaging:
  run `npm run build:wasm:node`, fix target-specific issues, and do not treat the work as complete until wasm packaging passes.
- Singleton bootstrap creates duplicates:
  add fixed-id existence checks before calling generated `Context::spawn(...)`.
- Event handler needs data that is not present in the graph payload or contexts:
  stop Rust patching and return to `graph-world` to fix topology/schema first.
- Runtime compiles but time is not advancing:
  verify `TimePlugin` is added and systems run on `Update` schedule.
- Rendering-related crate unexpectedly appears:
  remove it and re-check `Cargo.toml` features/dependencies.
- `wasm-pack` build fails due to missing wasm target:
  run `rustup target add wasm32-unknown-unknown` and retry.
- `wasm-pack` build fails because crate type is not compatible:
  add `[lib] crate-type = ["cdylib", "rlib"]` to root `Cargo.toml`.
- Android build fails with NDK not found:
  set `ANDROID_NDK_HOME` and `ANDROID_NDK_ROOT`, then rerun `cargo ndk` scripts.
- Windows/Linux cross compile fails on macOS linker:
  install `zig` and use `cargo zigbuild` scripts (do not use plain `cargo build` for those targets).
- iOS build fails due to missing target:
  run `rustup target add <ios-target>` and retry.
- Request needs graph topology change:
  pause Rust edits and switch back to `graph-world` first.
- `EventReader<T>` or `EventWriter<T>` found in code:
  replace with `On<T>` trigger + `app.add_observer(...)`. `EventReader`/`EventWriter` are legacy patterns incompatible with the observer-based event dispatch used by `world-bevy`.

## Simulator Log Verification Loop

When implementing or fixing Systems/EventSystems, use the simulator API to verify runtime behavior in a tight iteration loop without restarting the GraphOS service.

### Verification flow: pause → inspect logs

When the simulator is running and you need to check runtime behavior:

1. **Pause** the simulator to freeze state at the current tick:
   - `POST /api/world/pause` — stops the clock, preserving all runtime state and accumulated logs.
2. **Inspect logs** via `GET /api/world/log` to verify expected `event`, `get`, `set`, `add`, and `del` records were produced up to the pause point.
3. **Analyze** the log entries against expected behavior. For each log entry, check:
   - Does the operation type (`event`/`get`/`set`/`add`/`del`) match what the System/EventSystem should produce?
   - Do the target entity/component IDs and values match expectations?
   - Is the ordering of operations correct relative to system execution order?
4. If logs are correct, `POST /api/world/resume` to continue. If incorrect, proceed to the fix cycle below.

**Why pause first:** pausing before log inspection ensures you're reading a stable snapshot at a known point in time, rather than chasing logs that are still being written. This is especially important for systems with fast tick rates or async event flows.

### Fix cycle: code change → reset → start

When logs show incorrect or missing behavior, apply the fix and restart the simulator without restarting the service:

1. **Modify code** — edit the Rust System/EventSystem code in `src/app/`.
2. **Rebuild wasm** — run `npm run build:wasm:node` to compile changes. The simulator picks up the new wasm automatically on next reset.
3. **Reset** — `POST /api/world/reset` clears all runtime state, entities, and accumulated logs, returning the simulator to initial state.
4. **Start** — `POST /api/world/start` starts the simulator clock from time zero, re-running bootstrap systems and initializing fresh state with the updated wasm.
5. **Drive scenario** — send events through `POST /api/world/event` to trigger the specific code path being verified (e.g. `{ "type": "MyEvent", "payload": { ... } }`).
6. **Pause and inspect** — `POST /api/world/pause` then `GET /api/world/log` to verify the fix.
7. If logs still show incorrect behavior, return to step 1 and repeat.

**Why reset-then-start (not just resume):** after a code change, the old runtime state, entities, and component values may be incompatible with the updated wasm logic. Reset ensures a clean slate; start re-runs all bootstrap/Startup systems with the new code so you're testing against fresh, consistent state. Skipping reset can produce false positives or crashes from stale entity data.

### API reference (from graph-world skill)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/world/state` | GET | Get simulator state: `{ duration, current, state, scale, fps }` |
| `/api/world/start` | POST | Start simulator clock from time zero |
| `/api/world/pause` | POST | Pause simulator clock, preserving state and logs for inspection |
| `/api/world/resume` | POST | Resume paused simulator from where it stopped |
| `/api/world/reset` | POST | Reset to initial state, clear all entities and logs |
| `/api/world/event` | POST | Send event payload into simulator |
| `/api/world/log` | GET | Query logged records (supports `startTime`/`endTime` params) |

### Key constraints

- Do not use `cargo run` for simulation log verification. `cargo run` launches a native binary that does not integrate with the GraphOS simulator API (`/api/world/*`). All log verification must go through the simulator API via `npm run build:wasm:node` + HTTP endpoints.
- Logs that must appear in `/api/world/log` should be emitted from runtime code with `commands.trigger(world_bevy::core::WorldLogEvent { log: ... })`; stdout logging is not a substitute for simulator records.
- Do not use the skill reference implementation's wasm for simulation. The simulator runs the project's own wasm built from the current workspace; verify via the project's `npm run build:wasm:node`, not by running the skill's `src/app/` reference code directly.
- Never restart the GraphOS service to pick up wasm changes; `npm run build:wasm:node` + `POST /api/world/reset` then `POST /api/world/start` is sufficient.
- After `npm run build:wasm:node` succeeds, immediately verify through the simulator API before declaring the fix complete.
- If the log payload is large, generate a focused analysis script rather than reading the full response manually (see graph-world skill for `/api/world/log` query patterns).
- Always pause before log inspection when the simulator is running; never read logs on a running simulator — the log tail may be incomplete or still being written.

## Example Requests

- Bootstrap a GraphOS world project for Rust runtime with no TypeScript runtime code.
- Initialize a Bevy headless app using only app/ecs/time and add a basic tick system in `src/app/game_system.rs`.
- Implement `WorldBootstrapSystem` and register it in `src/app/mod.rs`.
- Implement a generated `StartGame` EventSystem in `src/app/start_game_system.rs`.
- Migrate existing world runtime from TS to Rust + Bevy ECS/time while keeping graph workflow unchanged.
