// #![feature(portable_simd)]
mod constants;
mod model;
mod token_output_stream;

use anyhow::{Error as E, Result};
use rand::{distributions::Distribution, SeedableRng};
use std::io::Write;
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;

const TEMPERATURE: Option<f64> = Some(0.7);

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
        let p = p.as_ref();
        let file = std::fs::File::open(p)?;
        let file_len = file.metadata()?.len();
        let expected_len = std::mem::size_of::<model::Weights>() as u64;
        if file_len != expected_len {
            anyhow::bail!("Unexpected length of file for {p:?}, {file_len} <> {expected_len}")
        }
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // the dodgy bit.
        let weights = unsafe { &*(mmap.as_ptr() as *const model::Weights) };
        Ok(Self { mmap, weights })
    }

    fn weights(&self) -> &model::Weights {
        self.weights
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let prompt = if args.len() < 2 { " ".to_string() } else { args[1].clone() };
    let mut state = model::State::<1>::new();
    let mmaped_weights = MmapedWeights::from_file(constants::MODEL_FILENAME)?;
    println!("state size:  {:4}MB", std::mem::size_of::<model::State<1>>() >> 20);
    println!("weight size: {:4}MB", std::mem::size_of::<model::Weights>() >> 20);
    let tokenizer = Tokenizer::from_file("tokenizer.json").map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    let mut lp = LogitsProcessor::new(299792458, TEMPERATURE);
    let eos_token = match tokenizer.get_token("<|endoftext|>") {
        Some(token) => token,
        None => anyhow::bail!("cannot find the </s> token"),
    };
    println!("processing prompt '{prompt}'");
    let prompt_tokens =
        tokenizer.tokenizer().encode(prompt, true).map_err(E::msg)?.get_ids().to_vec();

    for &t in prompt_tokens.iter() {
        state.update(&[t], mmaped_weights.weights());
        if let Some(t) = tokenizer.next_token(t)? {
            print!("{t}")
        }
    }
    std::io::stdout().flush()?;

    let start_gen = std::time::Instant::now();
    let mut generated_tokens = 0usize;
    loop {
        let next_token = lp.sample(&state.logits()[0])? as u32;
        if next_token == eos_token {
            println!();
            break;
        }
        if let Some(t) = tokenizer.next_token(next_token as u32)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        state.update(&[next_token], mmaped_weights.weights());
        generated_tokens += 1;
    }
    let dt = start_gen.elapsed();
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>) -> Self {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) { None } else { temperature };
        Self { rng: rand::rngs::StdRng::seed_from_u64(seed), temperature }
    }

    fn sample_argmax(&mut self, logits: &[f32]) -> Result<u32> {
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    pub fn sample(&mut self, logits: &[f32; constants::VOCAB_SIZE]) -> Result<u32> {
        let next_token = match self.temperature {
            None => self.sample_argmax(logits)?,
            Some(temperature) => {
                let max_logit = logits.iter().max_by(|f1, f2| f1.total_cmp(f2)).unwrap();
                let mut prs = [0f32; constants::VOCAB_SIZE];
                let mut sum_pr = 0f32;
                for (pr, logit) in prs.iter_mut().zip(logits.iter()) {
                    *pr = ((logit - max_logit) / temperature as f32).exp();
                    sum_pr += *pr;
                }
                for pr in prs.iter_mut() {
                    *pr /= sum_pr
                }
                self.sample_multinomial(&prs)?
            }
        };
        Ok(next_token)
    }
}
