use crate::r#gen::world::WorldContext;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

pub mod game_system;
pub mod start_game_system;
pub mod world_bootstrap_system;

pub fn reg(app: &mut App) {
    app.add_observer(on_world_spawn);
    world_bootstrap_system::reg(app);
    game_system::reg(app);
    start_game_system::reg(app);
}

fn on_world_spawn(trigger: On<world_bevy::core::Startup>, mut commands: Commands) {
    WorldContext::spawn(&mut commands, None, "World");
}
