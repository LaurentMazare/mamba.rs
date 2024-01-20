// https://huggingface.co/state-spaces/mamba-130m/blob/main/config.json
pub const D_MODEL: usize = 768;
pub const N_LAYER: usize = 24;
pub const VOCAB_SIZE_: usize = 50277;
pub const PAD_VOCAB_SIZE_MULTIPLE: usize = 8;

pub const D_CONV: usize = 4;
pub const D_STATE: usize = 16;

pub const D_INNER: usize = D_MODEL * 2;
pub const DT_RANK: usize = (D_MODEL + 15) / 16;
pub const VOCAB_SIZE: usize =
    (VOCAB_SIZE_ + PAD_VOCAB_SIZE_MULTIPLE - 1) / PAD_VOCAB_SIZE_MULTIPLE * PAD_VOCAB_SIZE_MULTIPLE;
