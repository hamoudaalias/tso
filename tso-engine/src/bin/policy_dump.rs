#![allow(dead_code, unused_assignments, unused_variables)]
/// ════════════════════════════════════════════════════════════════════════════
///  Diagnostic de convergence : politique apprise vs optimale en S1
///
///  Routine :
///    1. Entraîne 500 épisodes en S1 (comme Phase 1c, avec ε annealing)
///    2. Logger par épisode : mean |δ|, V(h) sur qq positions, reward total
///    3. Test ε=0 : dump la politique gloutonne pour chaque position de la grille
///    4. Compare à la politique optimale (descente de gradient BFS vers eau)
///
///  Résultat attendu :
///    - Si la politique apprise est cohérente (ex: va vers l'eau) mais
///      sous-optimale → le cervelet converge partiellement (problème
///      d'efficacité, pas de convergence brisée)
///    - Si la politique est quasi-aléatoire ou bloquée sur une seule action
///      → la convergence est cassée (le TD-error n'a pas propagé
///      correctement le gradient)
/// ════════════════════════════════════════════════════════════════════════════

use ndarray::Array1;
use rand::Rng;
use std::time::Instant;
use tso_engine::tso_engine::TsoEngine;

const W: usize = 5; const H: usize = 5;
const PERCEPTION_DIM: usize = 6; const N_ACTIONS: usize = 4; const MAX_STEPS: usize = 150;
const WATER_POSITIONS: [(usize, usize); 3] = [(1, 1), (3, 3), (1, 4)];
const FOOD_POSITIONS: [(usize, usize); 0] = [];

const TRAIN_EPS: usize = 500;
const TEST_EPS: usize = 100;

// ─── Environnement ──────────────────────────────────────────────────────────
struct GridEnv5x5 { agent: (usize, usize), step: usize, done: bool }
impl GridEnv5x5 {
    fn new() -> Self { GridEnv5x5 { agent: (2,2), step:0, done:false } }
    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(0..W); let y = rng.gen_range(0..H);
            if !WATER_POSITIONS.contains(&(x,y)) && !FOOD_POSITIONS.contains(&(x,y)) {
                self.agent=(x,y); break;
            }
        }
        self.step=0; self.done=false;
    }
    fn perceive(&self) -> Array1<f64> {
        let (x,y)=self.agent; let ix=x as isize; let iy=y as isize;
        let ray=|dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let mut fs=0.0; for &(fx,fy)in&FOOD_POSITIONS{
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
        let mut ws=0.0; for &(wx,wy)in&WATER_POSITIONS{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws])
    }
    fn step_env(&mut self, action: usize) -> f64 {
        if self.done{return 0.0;} self.step+=1;
        let(dx,dy)=match action{0=>(0,-1),1=>(0,1),2=>(-1,0),3=>(1,0),_=>(0,0)};
        let nx=self.agent.0 as isize+dx; let ny=self.agent.1 as isize+dy;
        if nx<0||ny<0||nx>=W as isize||ny>=H as isize{
            if self.step>=MAX_STEPS{self.done=true;}return-0.5;}
        self.agent=(nx as usize,ny as usize);
        if WATER_POSITIONS.contains(&self.agent){self.done=true;return 10.0;}
        if FOOD_POSITIONS.contains(&self.agent){self.done=true;return 10.0;}
        if self.step>=MAX_STEPS{self.done=true;return-1.0;}
        -0.02
    }
}

fn compute_bfs_potential() -> Vec<Vec<f64>> {
    use std::collections::VecDeque;
    let d_max=((W-1)+(H-1))as f64;let mut pot=vec![vec![0.0;H];W];
    let mut dist=vec![vec![None::<usize>;H];W];let mut q=VecDeque::new();
    for &(wx,wy)in&WATER_POSITIONS{dist[wx][wy]=Some(0);q.push_back((wx,wy));}
    while let Some((cx,cy))=q.pop_front(){let d=dist[cx][cy].unwrap();
        for(dx,dy)in[(0,1),(0,-1),(1,0),(-1,0)]{let nx=cx as isize+dx;let ny=cy as isize+dy;
            if nx>=0&&ny>=0&&nx<W as isize&&ny<H as isize{let(nx,ny)=(nx as usize,ny as usize);
                if dist[nx][ny].is_none(){dist[nx][ny]=Some(d+1);q.push_back((nx,ny));}}}}
    for x in 0..W{for y in 0..H{pot[x][y]=match dist[x][y]{Some(d)=>-2.5*d as f64/d_max,None=>-2.5};}}
    pot
}

/// Politique optimale : suivre le gradient de Φ_BFS (vers l'eau)
fn optimal_policy(bfs_pot: &[Vec<f64>]) -> [[usize; H]; W] {
    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y] = 4; continue; } // NOP
        let mut best_a = 0; let mut best_v = bfs_pot[x][y];
        for (a, (dx, dy)) in [(0,-1),(0,1),(-1,0),(1,0)].iter().enumerate() {
            let nx = x as isize + dx; let ny = y as isize + dy;
            if nx >= 0 && ny >= 0 && nx < W as isize && ny < H as isize {
                let v = bfs_pot[nx as usize][ny as usize];
                if v > best_v { best_v = v; best_a = a; }
            }
        }
        policy[x][y] = best_a;
    }}
    policy
}

/// Simule un test ε=0 sur une position donnée. Retourne l'action choisie.
fn get_greedy_action(engine: &mut TsoEngine, pos: (usize, usize), bfs_pot: &[Vec<f64>]) -> usize {
    let mut env = GridEnv5x5::new();
    env.agent = pos;
    let p = env.perceive();
    engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
    engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;
    let bv = Some(bfs_pot[pos.0][pos.1]);
    // On appelle step avec ε=0 (pas de noise), ce qui produit l'action gloutonne
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;
    engine.step(&p, 0.0, bv, &[])
}

#[derive(Clone)]
struct PolicyDump {
    policy: [[usize; H]; W],      // action choisie par position
    values: [[f64; H]; W],        // V(h) approximée par le critic
}

fn dump_policy(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>]) -> PolicyDump {
    // On réinitialise l'engine entre les dump pour éviter la contamination
    // (l'appel à step modifie l'état interne)
    let mut policy = [[0usize; H]; W];
    let mut values = [[0.0; H]; W];

    let original_eps = engine.cerebellum.epsilon;
    let original_noise = engine.cerebellum.noise_std;
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) {
            policy[x][y] = 4; // marqueur spécial
            continue;
        }
        let action = get_greedy_action(engine, (x,y), bfs_pot);
        policy[x][y] = action;
        // L'engine a maintenant forward_logits en cache → V(h) est dispo
        // On la récupère... via predict_value (publique ?)
        // predict_value n'est pas pub. Mais on peut la contourner.
        // Pour l'instant on stocke 0 et on regarde les logits plus tard.
        values[x][y] = 0.0;
        engine.end_episode(); // reset trace entre chaque position
    }}

    engine.cerebellum.epsilon = original_eps;
    engine.cerebellum.noise_std = original_noise;
    PolicyDump { policy, values }
}

fn print_policy_grid(policy: &[[usize; H]; W], title: &str) {
    eprintln!("── {} ──", title);
    eprintln!("        Col 0  Col 1  Col 2  Col 3  Col 4");
    for y in 0..H {
        eprint!("  Row {}  ", y);
        for x in 0..W {
            let c = match policy[x][y] {
                0 => 'N', 1 => 'S', 2 => 'W', 3 => 'E',
                4 => '░', // water/terminal
                _ => '?',
            };
            eprint!("   {}   ", c);
        }
        eprintln!();
    }
    eprintln!();
}

fn run_train_ep(engine: &mut TsoEngine, bfs_pot: &[Vec<f64>], _ep: usize, _is_training: bool) -> (f64, bool, f64) {
    let mut env = GridEnv5x5::new();
    env.reset();
    engine.end_episode();

    engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
    engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;

    let mut total = 0.0; let mut succeeded = false;
    let _sum_delta = 0.0; let _delta_count = 0u64;
    // On utilise le champ debug_step_dump pour capturer |δ|
    // Mais le δ n'est pas exposé. On va logger via l'approximation :
    // la reward totale moins la dernière V(h) donne une idée de la TD-error.
    // Pour l'instant on se contente du reward total.

    let p = env.perceive();
    let bv = Some(bfs_pot[env.agent.0][env.agent.1]);
    let mut a = engine.step(&p, 0.0, bv, &[]);

    while !env.done {
        let r = env.step_env(a); total += r;
        if env.done { succeeded = r > 0.0;
            let pt = env.perceive(); engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]); break; }
        engine.hypothalamus.energy = 1.0; engine.hypothalamus.hydration = 1.0;
        engine.hypothalamus.temperature = 0.5; engine.hypothalamus.sleep_debt = 0.0;
        let pt = env.perceive();
        a = engine.step(&pt, r, Some(bfs_pot[env.agent.0][env.agent.1]), &[]);
    }
    engine.end_episode();
    if engine.cerebellum.replay.len() >= 64 {
        engine.cerebellum.replay_train(64, 0.95, 10);
    }
    (total, succeeded, 0.0)
}

fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  DIAGNOSTIC : Politique apprise en S1 vs optimale                   ║");
    eprintln!("║  Grille 5×5, eau en (1,1),(3,3),(1,4), dim=6 hd=4                   ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  On entraîne 500 épisodes S1, puis on dump la politique ε=0          ║");
    eprintln!("║  pour chaque position de la grille.                                  ║");
    eprintln!("║                                                                       ║");
    eprintln!("║  Si la politique apprend des patterns non-triviaux :                  ║");
    eprintln!("║    convergence partielle, le problème est l'efficacité.               ║");
    eprintln!("║  Si la politique est uniforme / aléatoire :                           ║");
    eprintln!("║    la convergence est cassée (poids n'ont pas appris).                ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    let bfs_pot = compute_bfs_potential();

    // Politique optimale (référence)
    let opt = optimal_policy(&bfs_pot);
    print_policy_grid(&opt, "POLITIQUE OPTIMALE (descente gradient BFS)");
    eprintln!("  Légende : N=0(Nord) S=1(Sud) W=2(Ouest) E=3(Est) ░=Eau\n");

    // ─── S1 : entraînement ───
    let mut engine = TsoEngine::with_hidden(PERCEPTION_DIM, N_ACTIONS, 4);
    engine.cerebellum.epsilon = 0.1;
    engine.cerebellum.noise_std = 0.1;
    engine.cerebellum.replay_lr = 0.05;
    engine.cerebellum.replay_only = false;
    engine.sleep_every_n_episodes = 0;
    engine.use_stationary_reward = true;

    let t0 = Instant::now();
    let mut train_rewards: Vec<f64> = Vec::with_capacity(TRAIN_EPS);
    let mut train_success: Vec<bool> = Vec::with_capacity(TRAIN_EPS);

    for ep in 1..=TRAIN_EPS {
        let remain = (TRAIN_EPS - ep).max(0) as f64 / TRAIN_EPS as f64;
        engine.cerebellum.epsilon = 0.8 * remain + 0.01;
        engine.cerebellum.noise_std = 0.3 * remain + 0.01;
        let (total, ok, _) = run_train_ep(&mut engine, &bfs_pot, ep, true);
        train_rewards.push(total); train_success.push(ok);

        if ep % 50 == 0 {
            eprintln!("  [entraînement] ép={}/{} avg_reward_last50={:.2} success_last50={:.1}% replay={} C={}",
                ep, TRAIN_EPS,
                train_rewards[ep-50..].iter().sum::<f64>() / 50.0,
                train_success[ep-50..].iter().filter(|&&s|s).count() as f64 / 50.0 * 100.0,
                engine.cerebellum.replay.len(),
                engine.num_concepts());
        }
    }

    let elapsed = t0.elapsed();
    let train_avg = train_rewards.iter().sum::<f64>() / TRAIN_EPS as f64;
    let train_last_200 = train_rewards[TRAIN_EPS-200..].iter().sum::<f64>() / 200.0;
    let train_success_rate = train_success.iter().filter(|&&s|s).count() as f64 / TRAIN_EPS as f64;

    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  ENTRAÎNEMENT TERMINÉ                                                ║");
    eprintln!("╠══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  {} épisodes en {:.1}s  avg={:.1}  last200={:.1}  success={:.1}%",
        TRAIN_EPS, elapsed.as_secs_f64(), train_avg, train_last_200, train_success_rate*100.0);
    eprintln!("║  Replay={}  Concepts={}  Edges={}  Φ={:.3}",
        engine.cerebellum.replay.len(), engine.num_concepts(),
        engine.graph.edges.len(), engine.current_phi);
    eprintln!("╚══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    // ─── TEST ε=0 : politique gloutonne ───
    engine.cerebellum.epsilon = 0.0;
    engine.cerebellum.noise_std = 0.0;

    let mut test_rewards: Vec<f64> = Vec::with_capacity(TEST_EPS);
    let mut test_success: Vec<bool> = Vec::with_capacity(TEST_EPS);
    for ep in 0..TEST_EPS {
        let (total, ok, _) = run_train_ep(&mut engine, &bfs_pot, 1000+ep, false);
        test_rewards.push(total); test_success.push(ok);
    }
    let test_avg = test_rewards.iter().sum::<f64>() / TEST_EPS as f64;
    let test_success_rate = test_success.iter().filter(|&&s|s).count() as f64 / TEST_EPS as f64;
    eprintln!("  TEST ε=0  avg={:.1}  success={:.1}%", test_avg, test_success_rate*100.0);
    eprintln!("  10 premiers tests: {:?}", &test_rewards[..10.min(test_rewards.len())]);
    eprintln!();

    // ─── DUMP : politique par position ───
    // Pour chaque position, on crée une copie propre de l'engine
    // (les poids sont les poids entraînés ; on veut juste l'action gloutonne)
    // Mais on ne peut pas cloner l'engine simplement. On va plutôt
    // créer un engine "lecture seule" avec les mêmes poids ?
    // Non, Rust ne permet pas ça simplement.
    //
    // Solution : on dump les logits en modifiant `forward_logits` pour qu'il
    // les retourne. Le Cerebellum::forward_logits est déjà public.
    // On va appeler forward_logits sur des perceptions synthétiques.

    // On prépare les perceptions pour chaque position
    let mut policy = [[0usize; H]; W];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { policy[x][y] = 4; continue; }
        // Simuler la perception à (x,y)
        let _env = GridEnv5x5::new(); // dummy
        let ix = x as isize; let iy = y as isize;
        let ray = |dx:isize,dy:isize|->f64{let mut d=0;let mut cx=ix+dx;let mut cy=iy+dy;
            while cx>=0&&cy>=0&&cx<W as isize&&cy<H as isize{d+=1;cx+=dx;cy+=dy;}
            d as f64/(W.max(H) as f64)};
        let mut fs=0.0; for &(fx,fy)in&FOOD_POSITIONS{
            let d=(((ix-fx as isize).abs().pow(2)+(iy-fy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{fs=(1.0-d/3.0).max(0.0);break;}}
        let mut ws=0.0; for &(wx,wy)in&WATER_POSITIONS{
            let d=(((ix-wx as isize).abs().pow(2)+(iy-wy as isize).abs().pow(2))as f64).sqrt();
            if d<=2.0{ws=(1.0-d/3.0).max(0.0);break;}}
        let p = Array1::from_vec(vec![ray(0,-1),ray(0,1),ray(-1,0),ray(1,0),fs,ws]);

        let logits = engine.cerebellum.forward_logits(&p);
        let best_action = logits.iter().enumerate()
            .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap())
            .map(|(i,_)| i).unwrap();
        policy[x][y] = best_action;
    }}

    print_policy_grid(&policy, "POLITIQUE APPRISE S1 (ε=0)");

    // Vérification : combien de positions correspondent à l'optimale ?
    let mut correct = 0; let mut total = 0;
    let mut action_counts = [0u64; 4];
    for x in 0..W { for y in 0..H {
        if WATER_POSITIONS.contains(&(x,y)) { continue; }
        total += 1;
        if policy[x][y] == opt[x][y] { correct += 1; }
        action_counts[policy[x][y]] += 1;
    }}
    eprintln!("  Précision vs optimale : {}/{} ({:.1}%)", correct, total, correct as f64/total as f64 * 100.0);
    eprintln!("  Distribution actions : N={} S={} W={} E={}",
        action_counts[0], action_counts[1], action_counts[2], action_counts[3]);
    eprintln!("  Aléatoire parfait : ~{:.1} par action", total as f64 / 4.0);

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  DIAGNOSTIC TERMINÉ                                                    ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}
