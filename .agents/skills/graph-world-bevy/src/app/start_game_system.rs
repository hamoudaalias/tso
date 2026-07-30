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
