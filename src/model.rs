#![allow(unused)]
use crate::constants::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct LinearNoBias<const IN: usize, const OUT: usize> {
    w: [[f32; IN]; OUT],
}

impl<const IN: usize, const OUT: usize> LinearNoBias<IN, OUT> {
    // https://github.com/srush/llama2.rs/blob/2ca8f3dc0d4aa945a29700271883af72d9043ef1/src/model.rs#L22
    pub fn forward<const B: usize>(self: &Self, xout: &mut [[f32; OUT]; B], x: &[[f32; IN]; B]) {
        for (xout, x) in xout.iter_mut().zip(x) {
            xout.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = self.w[i].iter().zip(x.iter()).fold(0.0, |acc, (&_w, &_x)| acc + _w * _x);
            });
        }
    }
}

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
            self.forward_one(outs, ins, 1e-5);
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

// See Figure 3, page 8, on https://arxiv.org/pdf/2312.00752.pdf
struct BlockWeights {
    norm: RmsNorm<D_MODEL>,
    in_proj1: LinearNoBias<D_MODEL, D_INNER>,
    in_proj2: LinearNoBias<D_MODEL, D_INNER>,
    x_proj: LinearNoBias<D_INNER, { DT_RANK + D_STATE * 2 }>,
    dt_proj: LinearNoBias<DT_RANK, D_INNER>,
    dt_proj_bias: [f32; D_INNER],
    out_proj: LinearNoBias<D_INNER, D_MODEL>,
}

pub struct Weights {
    embedding: [[f32; D_MODEL]; VOCAB_SIZE],
    layers: [BlockWeights; N_LAYER],
    norm_f: RmsNorm<D_MODEL>,
    lm_head: LinearNoBias<D_MODEL, VOCAB_SIZE>,
}

pub struct State<const B: usize> {
    xs: [[f32; D_MODEL]; B],
    norm_xs: [[f32; D_MODEL]; B],
    logits: [[f32; VOCAB_SIZE]; B],
    proj_for_conv: [[f32; D_INNER]; B],
    proj_for_silu: [[f32; D_INNER]; B],
}

impl<const B: usize> State<B> {
    pub fn new() -> Self {
        Self {
            xs: [[0f32; D_MODEL]; B],
            norm_xs: [[0f32; D_MODEL]; B],
            logits: [[0f32; VOCAB_SIZE]; B],
            proj_for_conv: [[0f32; D_INNER]; B],
            proj_for_silu: [[0f32; D_INNER]; B],
        }
    }

    pub fn update(&mut self, tokens: &[usize; B], w: &Weights) {
        for (xs, token) in self.xs.iter_mut().zip(tokens) {
            xs.copy_from_slice(&w.embedding[*token]);
        }

        for layer in w.layers.iter() {
            layer.norm.forward(&mut self.norm_xs, &self.xs, 1e-5);

            {
                // Mixer forward.
                layer.in_proj1.forward(&mut self.proj_for_conv, &self.norm_xs);
                layer.in_proj2.forward(&mut self.proj_for_silu, &self.norm_xs);

                // TODO: conv1d
                for s in self.proj_for_conv.iter_mut() {
                    silu_in_place(s)
                }
                // TODO: ssm

                for (s_out, s_in) in self.proj_for_silu.iter_mut().zip(self.proj_for_conv.iter()) {
                    silu_in_place(s_out);
                    mul_in_place(s_out, s_in);
                }
                // Put the result back in norm_xs
                layer.out_proj.forward(&mut self.norm_xs, &self.proj_for_silu)
            }

            // Residual connections
            for (norm_xs, xs) in self.norm_xs.iter().zip(self.xs.iter_mut()) {
                add_in_place(xs, norm_xs)
            }
        }

        w.norm_f.forward(&mut self.norm_xs, &self.xs, 1e-5);
        w.lm_head.forward(&mut self.logits, &self.norm_xs)
    }

    pub fn logits(&self) -> &[[f32; VOCAB_SIZE]; B] {
        &self.logits
    }
}
