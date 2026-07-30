use tso_engine::terrarium::Terrarium;

#[test]
fn test_terrarium_default_perishable() {
    let t = Terrarium::new(0);
    assert!(t.perishable);
}

#[test]
fn test_terrarium_resource_consumed_on_visit() {
    let mut t = Terrarium::new(1);
    // Food at (2,1). Start at (1,1), move E (action 3) → (2,1)
    t.agent = (1, 1);
    let r = t.step(3);
    assert!(r >= 10.0, "First food visit should give reward (got {r})");
    t.agent = (1, 1);
    let r2 = t.step(3);
    assert!(r2 < 10.0, "Second visit to same spot should not give food (got {r2})");
}

#[test]
fn test_terrarium_water_consumed_and_respawns() {
    let mut t = Terrarium::new(4);
    // Water at (5,1). Start at (4,1), move E (action 3) → (5,1)
    t.agent = (4, 1);
    let r = t.step(3);
    assert!(r >= 8.0, "First water visit should give reward (got {r})");
    t.agent = (4, 1);
    let r2 = t.step(3);
    assert!(r2 < 8.0, "Second visit should not give water (got {r2})");
}

#[test]
fn test_terrarium_energy_decay() {
    let mut t = Terrarium::new(3);
    t.energy = 0.1;
    t.agent = (1, 1);
    t.step(0);
    assert!(t.energy < 0.1, "Energy should decay each step");
}

#[test]
fn test_terrarium_non_perishable_behavior_unchanged() {
    let mut t = Terrarium::new(5);
    t.perishable = false;
    t.agent = (1, 1);
    let r = t.step(3);
    assert!(r >= 10.0, "First food (got {r})");
    t.agent = (1, 1);
    let r2 = t.step(3);
    assert!(r2 >= 10.0, "Non-perishable: food should persist (got {r2})");
}

#[test]
fn test_terrarium_food_sensed_drops_after_consumption() {
    let mut t = Terrarium::new(6);
    t.agent = (1, 1);
    let p_before = t.perception(None);
    let food_near_before = p_before[4];
    assert!(food_near_before > 0.0, "Food should be sensed before visit");
    t.step(3);
    let p_after = t.perception(None);
    let food_near_after = p_after[4];
    assert!(food_near_after < food_near_before, "Food sensing should drop after consumption");
}

#[test]
fn test_terrarium_steps_dont_panic() {
    let mut t = Terrarium::new(7);
    for _ in 0..10 {
        let a = rand::random::<usize>() % 4;
        t.step(a);
    }
}
