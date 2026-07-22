use std::collections::VecDeque;

pub struct Episode {
    pub sequence: Vec<usize>,
}

pub struct EpisodicMemory {
    episodes: Vec<Episode>,
    #[allow(dead_code)]
    max_len: usize,
}

impl EpisodicMemory {
    pub fn new(max_episode_len: usize) -> Self {
        EpisodicMemory {
            episodes: Vec::new(),
            max_len: max_episode_len,
        }
    }

    pub fn store(&mut self, sequence: &[usize]) {
        let seq: Vec<usize> = sequence.iter().copied().collect();
        self.episodes.push(Episode { sequence: seq });
    }

    pub fn recall(&self, context: &[usize]) -> Option<usize> {
        if context.is_empty() {
            return None;
        }
        let mut best_len = 0usize;
        let mut best_next = None;

        for ep in &self.episodes {
            if ep.sequence.len() < 2 {
                continue;
            }
            let max_check = context.len().min(ep.sequence.len() - 1);
            for match_len in (1..=max_check).rev() {
                let ctx_suffix = &context[context.len() - match_len..];
                let ep_prefix = &ep.sequence[..match_len];
                if ctx_suffix == ep_prefix {
                    if match_len > best_len {
                        best_len = match_len;
                        best_next = Some(ep.sequence[match_len]);
                    }
                    break;
                }
            }
        }

        best_next
    }
}

pub struct ContextBuffer {
    buffer: VecDeque<usize>,
    max_len: usize,
}

impl ContextBuffer {
    pub fn new(max_len: usize) -> Self {
        ContextBuffer {
            buffer: VecDeque::new(),
            max_len,
        }
    }

    pub fn push(&mut self, word: usize) {
        if self.buffer.len() >= self.max_len {
            self.buffer.pop_front();
        }
        self.buffer.push_back(word);
    }

    pub fn as_slice(&self) -> Vec<usize> {
        self.buffer.iter().copied().collect()
    }
}
