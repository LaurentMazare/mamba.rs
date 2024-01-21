#![allow(clippy::needless_range_loop)]
use crate::constants::*;
use rayon::prelude::*;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LinearNoBias<const IN: usize, const OUT: usize> {
    w: [[f32; IN]; OUT],
}

impl<const IN: usize, const OUT: usize> LinearNoBias<IN, OUT> {
    // https://github.com/srush/llama2.rs/blob/2ca8f3dc0d4aa945a29700271883af72d9043ef1/src/model.rs#L22
    pub fn forward<const B: usize>(&self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) {
        for (xout, x) in xout.iter_mut().zip(x) {
            xout.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = self.w[i].iter().zip(x.iter()).fold(0.0, |acc, (&_w, &_x)| acc + _w * _x);
            });
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct RmsNorm<const DIM: usize> {
    w: [f32; DIM],
}

impl<const DIM: usize> RmsNorm<DIM> {
    fn forward_one(&self, o: &mut [f32; DIM], xo: &[f32; DIM], epsilon: f32) {
        // calculate sum of squares
        let mut ss = xo.iter().fold(0.0, |acc, x| acc + x * x);

        // take mean
        ss /= DIM as f32;
        ss += epsilon;
        ss = 1.0 / ss.sqrt();
        // normalize and scale
        for (j, weight_j) in self.w.iter().enumerate() {
            // Solve some borrow nonsense.
            o[j] = weight_j * ss * xo[j];
        }
    }

    fn forward<const B: usize>(
        &self,
        outs: &mut [[f32; DIM]; B],
        ins: &[[f32; DIM]; B],
        epsilon: f32,
    ) {
        for (outs, ins) in outs.iter_mut().zip(ins.iter()) {
            self.forward_one(outs, ins, epsilon);
        }
    }
}

fn add_in_place(a: &mut [f32], b: &[f32]) {
    for (a_i, b_i) in a.iter_mut().zip(b) {
        *a_i += b_i;
    }
}

fn mul_in_place(a: &mut [f32], b: &[f32]) {
    for (a_i, b_i) in a.iter_mut().zip(b) {
        *a_i *= b_i;
    }
}

fn silu_in_place(s: &mut [f32]) {
    for s in s.iter_mut() {
        *s = *s * (1.0 / (1.0 + (-*s).exp()));
    }
}

fn softplus_in_place(s: &mut [f32]) {
    // No softplus threshold here...
    for s in s.iter_mut() {
        *s = (s.exp() + 1.).ln()
    }
}

fn dot<const B: usize>(v1: &[f32; B], v2: &[f32; B]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&v1, &v2)| v1 * v2).sum::<f32>()
}

pub trait ModelWeights {
    type State<const B: usize>;
    const MODEL_FILENAME: &'static str;

    fn new_state<const B: usize>() -> Self::State<B>;
    fn update_state<const B: usize>(&self, state: &mut Self::State<B>, tokens: &[u32; B]);
    fn state_logits<const B: usize>(state: &Self::State<B>) -> &[[f32; VOCAB_SIZE]; B];
}

pub mod model_130m {
    use super::*;
    pub use params_130m::*;
    include!("model_inc.rs");
}

pub mod model_370m {
    use super::*;
    pub use params_370m::*;
    include!("model_inc.rs");
}

pub mod model_790m {
    use super::*;
    pub use params_790m::*;
    include!("model_inc.rs");
}

pub mod model_1_4b {
    use super::*;
    pub use params_1_4b::*;
    include!("model_inc.rs");
}

pub mod model_2_8b {
    use super::*;
    pub use params_2_8b::*;
    include!("model_inc.rs");
}
