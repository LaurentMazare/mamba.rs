mod constants;
mod model;

use anyhow::Result;

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
    let mut next_token = 209;
    for i in 0..40 {
        state.update(&[next_token], mmaped_weights.weights());
        next_token = argmax(&state.logits()[0]).unwrap();
        println!("{i} {next_token}");
    }
    // println!("{:?}", state.logits());
    Ok(())
}
