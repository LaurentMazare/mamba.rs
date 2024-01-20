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

#[repr(C)]
struct BlockWeights {
    norm: RmsNorm<D_MODEL>,
    in_proj1: LinearNoBias<D_MODEL, D_INNER>,
    in_proj2: LinearNoBias<D_MODEL, D_INNER>,
    x_proj1: LinearNoBias<D_INNER, DT_RANK>,
    x_proj2: LinearNoBias<D_INNER, D_STATE>,
    x_proj3: LinearNoBias<D_INNER, D_STATE>,
    dt_proj: LinearNoBias<DT_RANK, D_INNER>,
    dt_proj_bias: [f32; D_INNER],
    out_proj: LinearNoBias<D_INNER, D_MODEL>,
    a: [[f32; D_STATE]; D_INNER],
    d: [f32; D_INNER],
    conv1d_weight: [[f32; D_INNER]; D_CONV],
    conv1d_bias: [f32; D_INNER],
}

#[repr(C)]
pub struct Weights {
    embedding: [[f32; D_MODEL]; VOCAB_SIZE],
    layers: [BlockWeights; N_LAYER],
    norm_f: RmsNorm<D_MODEL>,
    lm_head: LinearNoBias<D_MODEL, VOCAB_SIZE>,
}

pub struct State<const B: usize> {
    // Persistent state
    hs: [[[[f32; D_STATE]; D_INNER]; B]; N_LAYER],
    prev_xs: [[[[f32; D_INNER]; B]; D_CONV]; N_LAYER],
    pos: usize,

    // Temporary variables, pre-allocated and only used in [update]
    xs: [[f32; D_MODEL]; B],
    norm_xs: [[f32; D_MODEL]; B],
    logits: [[f32; VOCAB_SIZE]; B],
    delta: [[f32; DT_RANK]; B],
    delta_proj: [[f32; D_INNER]; B],
    b: [[f32; D_STATE]; B],
    c: [[f32; D_STATE]; B],
    proj_for_conv: [[f32; D_INNER]; B],
    proj_for_silu: [[f32; D_INNER]; B],
}

impl<const B: usize> State<B> {
    pub fn new() -> Self {
        Self {
            hs: [[[[0f32; D_STATE]; D_INNER]; B]; N_LAYER],
            prev_xs: [[[[0f32; D_INNER]; B]; D_CONV]; N_LAYER],
            pos: 0,

            xs: [[0f32; D_MODEL]; B],
            norm_xs: [[0f32; D_MODEL]; B],
            logits: [[0f32; VOCAB_SIZE]; B],
            delta: [[0f32; DT_RANK]; B],
            delta_proj: [[0f32; D_INNER]; B],
            b: [[0f32; D_STATE]; B],
            c: [[0f32; D_STATE]; B],
            proj_for_conv: [[0f32; D_INNER]; B],
            proj_for_silu: [[0f32; D_INNER]; B],
        }
    }

    pub fn update(&mut self, tokens: &[usize; B], w: &Weights) {
        for (xs, token) in self.xs.iter_mut().zip(tokens) {
            xs.copy_from_slice(&w.embedding[*token]);
        }

        // See Figure 3, page 8, on https://arxiv.org/pdf/2312.00752.pdf
        for ((layer, hs), prev_xs) in
            w.layers.iter().zip(self.hs.iter_mut()).zip(self.prev_xs.iter_mut())
        {
            layer.norm.forward(&mut self.norm_xs, &self.xs, 1e-5);

            {
                // Mixer forward.
                layer.in_proj1.forward(&mut self.proj_for_conv, &self.norm_xs);
                layer.in_proj2.forward(&mut self.proj_for_silu, &self.norm_xs);

                let pos = self.pos % D_STATE;
                for b in 0..B {
                    prev_xs[pos % D_CONV][b].copy_from_slice(&self.proj_for_conv[b])
                }
                // Apply the conv1d and put the result in proj_for_conv.
                for (b, proj_for_conv) in self.proj_for_conv.iter_mut().enumerate() {
                    proj_for_conv.copy_from_slice(&layer.conv1d_bias);
                    for d_c in 0..D_CONV {
                        let pos_minus_d_c = (pos + D_CONV - d_c) % D_CONV;
                        for d_i in 0..D_INNER {
                            proj_for_conv[d_i] =
                                layer.conv1d_weight[d_c][d_i] * prev_xs[pos_minus_d_c][b][d_i]
                        }
                    }
                }

                for s in self.proj_for_conv.iter_mut() {
                    silu_in_place(s)
                }
                {
                    // SSM + Selection, we're doing inference here so only need the last step of
                    // the sequence.
                    // Algorithm 3.2 on page 6, https://arxiv.org/pdf/2312.00752.pdf
                    layer.x_proj1.forward(&mut self.delta, &self.proj_for_conv);
                    layer.x_proj2.forward(&mut self.b, &self.proj_for_conv);
                    layer.x_proj3.forward(&mut self.c, &self.proj_for_conv);

                    // Weird, what isn't this multiplication combined with x_proj1?
                    layer.dt_proj.forward(&mut self.delta_proj, &self.delta);
                    for delta_proj in self.delta_proj.iter_mut() {
                        add_in_place(delta_proj, &layer.dt_proj_bias)
                    }
                    for s in self.delta_proj.iter_mut() {
                        softplus_in_place(s);
                    }

                    // Selective scan part
                    for b in 0..B {
                        // Eqn (2a), page 3, h_t = Ab h_{t-1} + Bb x_t
                        for d_i in 0..D_INNER {
                            let delta = self.delta_proj[b][d_i];
                            let x = self.proj_for_conv[b][d_i];
                            for d_s in 0..D_STATE {
                                let a = layer.a[d_i][d_s];
                                let b_ = self.b[b][d_s];
                                hs[b][d_i][d_s] =
                                    hs[b][d_i][d_s] * (delta * a).exp() + delta * b_ * x;
                            }
                        }
                    }
                    // Put the result back in proj_for_conv
                    // y_t = c * h_t
                    for b in 0..B {
                        for d_i in 0..D_INNER {
                            self.proj_for_conv[b][d_i] = dot(&self.c[b], &hs[b][d_i])
                                + layer.d[d_i] * self.proj_for_conv[b][d_i]
                        }
                    }
                }

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
        self.pos += 1;

        w.norm_f.forward(&mut self.norm_xs, &self.xs, 1e-5);
        w.lm_head.forward(&mut self.logits, &self.norm_xs)
    }

    pub fn logits(&self) -> &[[f32; VOCAB_SIZE]; B] {
        &self.logits
    }
}
