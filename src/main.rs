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

fn main() -> Result<()> {
    println!("starting...");
    let mut state = model::State::<1>::new();
    let mmaped_weights = MmapedWeights::from_file("mamba-130m.bin")?;
    state.update(&[0], mmaped_weights.weights());
    println!("{:?}", state.logits());
    Ok(())
}
