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
        // Hook GameContext spawn-side initialization here when needed.
    }
}

fn on_despawn_system(removed: RemovedComponents<GameContext>) {
    for _entity in removed.read() {
        // Hook GameContext cleanup here when needed.
    }
}
