use std::path::Path;

pub enum TSOTokenizer {
    HuggingFace(tokenizers::Tokenizer),
    Whitespace {
        word_to_id: std::collections::HashMap<String, u32>,
        next_id: u32,
    },
}

impl TSOTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let t = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self::HuggingFace(t))
    }

    pub fn whitespace() -> Self {
        Self::Whitespace {
            word_to_id: std::collections::HashMap::new(),
            next_id: 0,
        }
    }

    pub fn encode(&mut self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        self.encode_with_words(text, add_special_tokens)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    pub fn encode_with_words(&mut self, text: &str, add_special_tokens: bool) -> Vec<(u32, String)> {
        match self {
            Self::HuggingFace(t) => t
                .encode(text, add_special_tokens)
                .map(|e| {
                    e.get_ids()
                        .iter()
                        .copied()
                        .map(|id| (id, format!("t{}", id)))
                        .collect()
                })
                .unwrap_or_default(),
            Self::Whitespace { word_to_id, next_id } => text
                .split_whitespace()
                .filter_map(|t| {
                    let word = t
                        .trim_matches(|c: char| c.is_ascii_punctuation())
                        .to_lowercase();
                    if word.is_empty() {
                        return None;
                    }
                    let id = *word_to_id.entry(word.clone()).or_insert_with(|| {
                        let id = *next_id;
                        *next_id += 1;
                        id
                    });
                    Some((id, word))
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let mut t = TSOTokenizer::whitespace();
        let ids = t.encode("hello world test", false);
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_empty_input() {
        let mut t = TSOTokenizer::whitespace();
        let ids = t.encode("", false);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_whitespace_consistent_ids() {
        let mut t = TSOTokenizer::whitespace();
        let a = t.encode("the cat", false);
        let b = t.encode("the dog", false);
        assert_eq!(a[0], b[0], "'the' should have same id across calls");
        assert_eq!(a, vec![0, 1]);
        assert_eq!(b, vec![0, 2]);
    }
}
