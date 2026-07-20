use std::path::Path;

use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::{AddedToken, Tokenizer};

/// Train a BPE tokenizer on raw texts and save it to disk.
pub fn train_bpe(
    texts: &[String],
    vocab_size: usize,
    min_frequency: u64,
    save_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let trainer = BpeTrainer::builder()
        .min_frequency(min_frequency)
        .vocab_size(vocab_size)
        .show_progress(false)
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
        ])
        .build();

    let mut trainer_wrapper = TrainerWrapper::BpeTrainer(trainer);
    let mut tokenizer = Tokenizer::new(BPE::default());
    // Override default ByteLevel pre-tokenizer with fast Whitespace
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace::default())));

    tokenizer
        .train(&mut trainer_wrapper, texts.iter().map(|s| s.as_str()))
        .map_err(|e| format!("BPE training failed: {}", e))?;

    tokenizer
        .save(save_path.as_ref().to_str().unwrap(), false)
        .map_err(|e| format!("BPE save failed: {}", e))?;

    Ok(())
}

/// Load a trained BPE tokenizer from disk.
pub fn load_bpe(
    path: impl AsRef<Path>,
) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    Tokenizer::from_file(path.as_ref().to_str().unwrap())
        .map_err(|e| format!("BPE load failed: {}", e).into())
}
