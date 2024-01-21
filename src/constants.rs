#![allow(unused)]
mod params_130m {
    // https://huggingface.co/state-spaces/mamba-130m/blob/main/config.json
    pub const D_MODEL: usize = 768;
    pub const N_LAYER: usize = 24;
    pub const MODEL_FILENAME: &str = "mamba-130m.bin";
}

mod params_370m {
    // https://huggingface.co/state-spaces/mamba-370m/blob/main/config.json
    pub const D_MODEL: usize = 1024;
    pub const N_LAYER: usize = 48;
    pub const MODEL_FILENAME: &str = "mamba-370m.bin";
}

mod params_790m {
    // https://huggingface.co/state-spaces/mamba-790m/blob/main/config.json
    pub const D_MODEL: usize = 1536;
    pub const N_LAYER: usize = 48;
    pub const MODEL_FILENAME: &str = "mamba-790m.bin";
}

mod params_1_4b {
    // https://huggingface.co/state-spaces/mamba-1.4b/blob/main/config.json
    pub const D_MODEL: usize = 1536;
    pub const N_LAYER: usize = 48;
    pub const MODEL_FILENAME: &str = "mamba-1_4b.bin";
}

mod params_2_8b {
    // https://huggingface.co/state-spaces/mamba-2.8b/blob/main/config.json
    pub const D_MODEL: usize = 2560;
    pub const N_LAYER: usize = 64;
    pub const MODEL_FILENAME: &str = "mamba-2_8b.bin";
}

#[cfg(feature = "370m")]
pub use params_370m::*;

#[cfg(feature = "790m")]
pub use params_790m::*;

#[cfg(feature = "1_4b")]
pub use params_1_4b::*;

#[cfg(feature = "2_8b")]
pub use params_2_8b::*;

#[cfg(not(any(feature = "370m", feature = "790m", feature = "1_4b", feature = "2_8b")))]
pub use params_130m::*;

pub const VOCAB_SIZE_: usize = 50277;
pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 8;
pub const VOCAB_SIZE: usize =
    (VOCAB_SIZE_ + PAD_VOCAB_SIZE_MULTIPLE - 1) / PAD_VOCAB_SIZE_MULTIPLE * PAD_VOCAB_SIZE_MULTIPLE;

pub const D_CONV: usize = 4;
pub const D_STATE: usize = 16;

pub const D_INNER: usize = D_MODEL * 2;
pub const DT_RANK: usize = (D_MODEL + 15) / 16;
