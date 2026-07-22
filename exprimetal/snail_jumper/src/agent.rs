use ndarray::Array1;
use tso_engine::neurons::LIFState;
use tso_engine::attractor::AttractorField;

use crate::game::GameState;

const EMBED_DIM: usize = 8;
const N_CLASSES: usize = 2;
const N_PROTOS: usize = 4;

pub struct TSOAgent {
    pub lif: LIFState,
    pub field: AttractorField,
    pub fitness: u64,
    pub alpha: f64,
}

impl TSOAgent {
    pub fn new() -> Self {
        let lif = LIFState::new(EMBED_DIM, 0.8);
        let field = AttractorField::new(EMBED_DIM, N_CLASSES, N_PROTOS, 0.05);
        TSOAgent { lif, field, fitness: 0, alpha: 0.8 }
    }

    pub fn from_field(field: AttractorField) -> Self {
        let lif = LIFState::new(EMBED_DIM, 0.8);
        TSOAgent { lif, field, fitness: 0, alpha: 0.8 }
    }

    pub fn encode_state(&self, game: &GameState) -> Array1<f64> {
        let mut v = Array1::zeros(EMBED_DIM);
        v[0] = game.player_x / game.screen_w;
        if let Some(obs) = game.nearest_obstacle() {
            v[1] = obs.x / game.screen_w;
            v[2] = obs.y / game.screen_h;
            v[3] = (obs.x - game.player_x) / game.screen_w;
        }
        v[4] = game.obstacles.len() as f64 / 10.0;
        for (i, obs) in game.obstacles.iter().enumerate() {
            if i < 3 {
                v[5 + i] = (obs.y / game.screen_h).min(1.0);
            }
        }
        v
    }

    pub fn decide(&mut self, game: &GameState) -> bool {
        let state = self.encode_state(game);
        self.lif.step(&state, false);

        let (class, _) = self.field.predict_with_distance(&self.lif.state);
        class == 0
    }

    pub fn learn(&mut self, game: &GameState) {
        let state = self.encode_state(game);
        if !game.alive {
            while self.field.n_classes() <= 1 { self.field.add_class(&state); }
            self.field.add_prototype(&state, 1);
        } else {
            while self.field.n_classes() == 0 { self.field.add_class(&state); }
            self.field.add_prototype(&state, 0);
        }
    }
}

impl Clone for TSOAgent {
    fn clone(&self) -> Self {
        TSOAgent {
            lif: LIFState::new(EMBED_DIM, self.alpha),
            field: self.field.clone(),
            fitness: 0,
            alpha: self.alpha,
        }
    }
}
