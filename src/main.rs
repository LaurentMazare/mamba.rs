#![feature(portable_simd)]
mod constants;
mod model;
mod tokenizer;

use anyhow::Result;
use std::io::Write;

// This struct is self-referential in a sense as if mmap gets dropped, weights would not be valid
// anymore.
struct MmapedWeights {
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    weights: &'static model::Weights,
}

impl MmapedWeights {
    /// This function is unsafe as it uses mmap and doesn't check the file size.
    fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let file = std::fs::File::open(p)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // the dodgy bit.
        let weights = unsafe { &*(mmap.as_ptr() as *const model::Weights) };
        Ok(Self { mmap, weights })
    }

    fn weights(&self) -> &model::Weights {
        self.weights
    }
}

fn argmax(v: &[f32]) -> Option<usize> {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|v| v.0)
}

fn main() -> Result<()> {
    println!("starting...");
    let mut state = model::State::<1>::new();
    let mmaped_weights = MmapedWeights::from_file("mamba-130m.bin")?;
    let tokenizer = tokenizer::Tokenizer::from_vocab_file("vocab.json")?;
    let mut next_token = 209;
    loop {
        state.update(&[next_token], mmaped_weights.weights());
        next_token = argmax(&state.logits()[0]).unwrap();

        // EOS is token-id 0.
        if next_token == 0 {
            println!();
            break;
        }
        let next_token = tokenizer.tokens(next_token)?;

        // Hacky decoding, the control characters are shifted by 256.
        for char in next_token.chars() {
            let c32 = char as u32;
            let char = if 256 <= c32 && c32 < 512 {
                char::from_u32(c32 - 256).unwrap_or(char)
            } else {
                char
            };
            print!("{char}");
        }
        std::io::stdout().flush()?;
    }
    Ok(())
}
