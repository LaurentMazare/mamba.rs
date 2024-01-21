// #![feature(portable_simd)]
mod constants;
mod model;
mod token_output_stream;

use anyhow::Result;
use std::io::Write;
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;

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
    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(anyhow::Error::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    let eos_token = match tokenizer.get_token("<|endoftext|>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the </s> token"),
    };
    let mut next_token = 209;
    let start_gen = std::time::Instant::now();
    let mut generated_tokens = 0usize;
    loop {
        state.update(&[next_token], mmaped_weights.weights());
        next_token = argmax(&state.logits()[0]).unwrap();
        generated_tokens += 1;

        if next_token == eos_token as usize {
            println!();
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token as u32)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    let dt = start_gen.elapsed();
    if let Some(rest) = tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
