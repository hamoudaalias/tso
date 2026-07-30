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
        if !existing_game.iter().any(|ctx| ctx.id == GAME_ID) {
            GameContext::spawn(
                &mut commands,
                Some(world.id.clone()),
                GAME_ID,
                GameTimeComponent(0.0),
                GameStateComponent(0),
            );
        }

        if !existing_scene.iter().any(|ctx| ctx.id == SCENE_ID) {
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
