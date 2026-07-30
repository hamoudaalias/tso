use tso_engine::episodic::{EpisodicMemory, ContextBuffer};

#[test]
fn test_new_is_empty() {
    let mem = EpisodicMemory::new(10);
    assert_eq!(mem.len(), 0);
}

#[test]
fn test_store_and_len() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[1, 2, 3]);
    assert_eq!(mem.len(), 1);
    mem.store(&[4, 5]);
    assert_eq!(mem.len(), 2);
}

#[test]
fn test_get_sequence() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[10, 20, 30, 40]);
    let seq = mem.get_sequence(0);
    assert!(seq.is_some());
    assert_eq!(seq.unwrap(), &[10, 20, 30, 40]);
}

#[test]
fn test_get_sequence_out_of_range() {
    let mem = EpisodicMemory::new(10);
    assert!(mem.get_sequence(0).is_none());
}

#[test]
fn test_recall_exact_match() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[1, 2, 3, 4]);
    let next = mem.recall(&[1, 2, 3]);
    assert_eq!(next, Some(4));
}

#[test]
fn test_recall_longest_match_wins() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[1, 2, 99]);
    mem.store(&[1, 2, 3, 4]);
    let next = mem.recall(&[1, 2, 3]);
    assert_eq!(next, Some(4));
}

#[test]
fn test_recall_no_match() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[1, 2, 3]);
    assert!(mem.recall(&[9, 9]).is_none());
}

#[test]
fn test_recall_empty_context() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[1, 2, 3]);
    assert!(mem.recall(&[]).is_none());
}

#[test]
fn test_recall_context_longer_than_episode() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[1, 2]);
    let next = mem.recall(&[0, 0, 1]);
    assert_eq!(next, Some(2));
}

#[test]
fn test_remap_basic() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[0, 1, 2]);
    mem.remap(&[Some(10), Some(20), Some(30)]);
    assert_eq!(mem.get_sequence(0).unwrap(), &[10, 20, 30]);
}

#[test]
fn test_remap_removes_dead_concepts() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[0, 1, 2]);
    mem.remap(&[Some(10), None, Some(30)]);
    assert_eq!(mem.get_sequence(0).unwrap(), &[10, 30]);
}

#[test]
fn test_remap_prunes_short_sequences() {
    let mut mem = EpisodicMemory::new(10);
    mem.store(&[0, 1]);
    mem.remap(&[Some(10), None]);
    assert_eq!(mem.len(), 0);
}

#[test]
fn test_context_buffer_new() {
    let cb = ContextBuffer::new(5);
    assert_eq!(cb.as_slice(), Vec::<usize>::new());
}

#[test]
fn test_context_buffer_push_and_slice() {
    let mut cb = ContextBuffer::new(5);
    cb.push(1);
    cb.push(2);
    cb.push(3);
    assert_eq!(cb.as_slice(), vec![1, 2, 3]);
}

#[test]
fn test_context_buffer_wraps_at_max() {
    let mut cb = ContextBuffer::new(3);
    cb.push(1);
    cb.push(2);
    cb.push(3);
    cb.push(4);
    assert_eq!(cb.as_slice(), vec![2, 3, 4]);
}

#[test]
fn test_context_buffer_remap() {
    let mut cb = ContextBuffer::new(5);
    cb.push(0);
    cb.push(1);
    cb.push(2);
    cb.remap(&[Some(10), None, Some(30)]);
    assert_eq!(cb.as_slice(), vec![10, 30]);
}
