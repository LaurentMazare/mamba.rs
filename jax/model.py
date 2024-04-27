from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
import math

# DTYPE_INNER = jax.dtypes.bfloat16
DTYPE_INNER = float


# https://github.com/state-spaces/mamba/blob/2a3704fd47ba817b415627b06fd796b971fdc137/mamba_ssm/models/mixer_seq_simple.py#L66
def embedding_init(key, shape, dtype):
    return jax.random.normal(key, shape, dtype=dtype) * 0.02


# https://github.com/state-spaces/mamba/blob/2a3704fd47ba817b415627b06fd796b971fdc137/mamba_ssm/models/mixer_seq_simple.py#L81
def kernel_init_prenorm(n_layer):
    # In the pytorch implementation, the non-linearity is leaky-relu with a=5**0.5.
    # calculate_gain -> sqrt(1/3)
    # return jax.random.he_uniform(key, shape, dtype=dtype)
    return jax.nn.initializers.variance_scaling(
        1.0 / (3 * n_layer) ** 0.5, "fan_in", "uniform"
    )


# https://github.com/state-spaces/mamba/blob/c7bca02c39909e88777c53ed910f78163d83c1ab/mamba_ssm/modules/mamba_simple.py#L83
def kernel_init_dt_proj(dt_rank):
    dt_init_std = dt_rank ** (-0.5)

    def init(key, shape, dtype):
        return (jax.random.uniform(key, shape, dtype=dtype) - 0.5) * (2 * dt_init_std)

    return init


def bias_init_dt_proj(d_inner):
    def init(key, shape, dtype):
        dt_max = 0.1
        dt_min = 0.001
        dt_init_floor = 1e-4

        dt = jax.random.uniform(key, shape, dtype=dtype)
        dt = dt * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt = jnp.exp(dt)
        dt = jnp.maximum(dt, dt_init_floor)
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        return inv_dt

    return init


# S4D real initialization.
# https://github.com/state-spaces/mamba/blob/86a3a902ca4189689aabf1c09174235024c7aede/mamba_ssm/modules/mamba_simple.py#L103
def a_log_init(rng, shape):
    d_inner, d_state = shape
    a = jnp.arange(1, 1 + d_state, dtype=float).reshape(1, d_state)
    a = jnp.broadcast_to(a, shape)
    return jnp.log(a)


class RMSNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, xs):
        dim = xs.shape[-1]
        weight = self.param("weight", jax.nn.initializers.constant(1.0), dim).reshape(
            1, 1, -1
        )

        orig_dtype = xs.dtype
        xs = xs.astype("float32")
        xs = xs * jax.lax.rsqrt(jnp.mean(xs**2, axis=-1, keepdims=True) + self.eps)
        return weight * xs.astype(orig_dtype)


@dataclass
class Config:
    d_model: int = 768
    n_layer: int = 24
    vocab_size_unpadded: int = 50277
    pad_vocab_size_multiple: int = 8
    d_conv: int = 4
    d_state: int = 16
    scan_op: str = "associative-scan"

    @property
    def vocab_size(self):
        pad = self.pad_vocab_size_multiple
        return (self.vocab_size_unpadded + pad - 1) // pad * pad

    @property
    def dt_rank(self):
        return (self.d_model + 15) // 16

    @property
    def d_inner(self):
        return self.d_model * 2


@partial(
    jax.checkpoint,
    policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
)
def selective_scan_p(u, delta, a, b, c, d):
    b_sz, seq_len, d_in = u.shape
    _d, n = a.shape
    delta = delta.reshape(b_sz, seq_len, d_in, 1)
    delta_a = delta * a.reshape(1, 1, d_in, n)
    delta_b_u = (
        delta * b.reshape(b_sz, seq_len, 1, n) * u.reshape(b_sz, seq_len, d_in, 1)
    )

    # (a, b) represents x -> a.x + b
    # (a, b) o (a', b') = (a.a', a.b' + b)

    @partial(
        jax.checkpoint,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    )
    def f(rhs, lhs):
        a = lhs["a"] + rhs["a"]
        b = jnp.exp(lhs["a"]) * rhs["b"] + lhs["b"]
        return {"a": a, "b": b}

    xs = jax.lax.associative_scan(
        f,
        {"a": delta_a, "b": delta_b_u},
        axis=1,
    )
    ys = jnp.einsum("bidn,bin->bid", xs["b"], c)
    return ys + u * d


def selective_scan_py(u, delta, a, b, c, d):
    b_sz, seq_len, d_in = u.shape
    _d, n = a.shape
    delta = delta.reshape(b_sz, seq_len, d_in, 1)
    delta_a = jnp.exp(delta * a.reshape(1, 1, d_in, n))
    delta_b_u = (
        delta * b.reshape(b_sz, seq_len, 1, n) * u.reshape(b_sz, seq_len, d_in, 1)
    )

    xs = jnp.zeros((b_sz, d_in, n))
    ys = []
    for i in range(seq_len):
        xs = delta_a[:, i] * xs + delta_b_u[:, i]
        ys_ = jnp.einsum("bdn,bn->bd", xs, c[:, i])
        ys.append(ys_)
    ys = jnp.stack(ys, axis=1)
    return ys + u * d


def selective_scan_s(u, delta, a, b, c, d):
    b_sz, seq_len, d_in = u.shape
    _d, n = a.shape
    delta = delta.reshape(b_sz, seq_len, d_in, 1)
    delta_a = jnp.exp(delta * a.reshape(1, 1, d_in, n))
    delta_b_u = (
        delta * b.reshape(b_sz, seq_len, 1, n) * u.reshape(b_sz, seq_len, d_in, 1)
    )

    xs = jnp.zeros((b_sz, d_in, n))

    def f(xs, delta_a_b_u):
        delta_a, delta_b_u = delta_a_b_u
        xs = delta_a * xs + delta_b_u
        return (xs, xs)

    _last_xs, ys = jax.lax.scan(
        f, xs, (delta_a.swapaxes(0, 1), delta_b_u.swapaxes(0, 1))
    )
    ys = jnp.einsum("ibdn,bin->bid", ys, c)
    return ys + u * d


class MambaBlock(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, xs):
        c = self.cfg
        in_proj = nn.Dense(c.d_inner * 2, use_bias=False, name="in_proj")
        out_proj = nn.Dense(
            c.d_model,
            use_bias=False,
            name="out_proj",
            kernel_init=kernel_init_prenorm(c.n_layer),
        )
        conv1d = nn.Conv(
            features=c.d_inner,
            kernel_size=(c.d_conv,),
            feature_group_count=c.d_inner,
            padding=c.d_conv - 1,
            use_bias=True,
            name="conv1d",
        )
        x_proj = nn.Dense(c.dt_rank + c.d_state * 2, use_bias=False, name="x_proj")
        dt_proj = nn.Dense(
            c.d_inner,
            use_bias=True,
            name="dt_proj",
            kernel_init=kernel_init_dt_proj(c.dt_rank),
            bias_init=bias_init_dt_proj(c.d_inner),
        )
        a_log = self.param("A_log", a_log_init, (c.d_inner, c.d_state))
        # https://github.com/state-spaces/mamba/blob/86a3a902ca4189689aabf1c09174235024c7aede/mamba_ssm/modules/mamba_simple.py#L114
        d = self.param("D", jax.nn.initializers.constant(1.0), (c.d_inner,))

        _b_sz, seq_len, _d_inner = xs.shape
        xs_and_res = in_proj(xs)
        xs = xs_and_res[:, :, : c.d_inner]
        res = xs_and_res[:, :, c.d_inner :]
        xs = conv1d(xs)
        xs = xs[:, :seq_len]
        xs = nn.activation.silu(xs)

        # SSM forward
        a = -jnp.exp(a_log)
        x_dbl = x_proj(xs)
        delta = x_dbl[..., : c.dt_rank]
        b = x_dbl[..., c.dt_rank : c.dt_rank + c.d_state]
        c = x_dbl[..., c.dt_rank + c.d_state :]

        delta = jax.nn.softplus(dt_proj(delta))
        if self.cfg.scan_op == "python":
            ss = selective_scan_py(xs, delta, a, b, c, d)
        elif self.cfg.scan_op == "scan":
            ss = selective_scan_s(xs, delta, a, b, c, d)
        elif self.cfg.scan_op == "associative-scan":
            ss = selective_scan_p(xs, delta, a, b, c, d)
        else:
            raise ValueError(f"unknown scan_op {self.cfg.scan_op}")
        ys = ss * nn.activation.silu(res)
        return out_proj(ys)


class ResidualBlock(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, xs):
        norm = RMSNorm(name="norm")
        mixer = MambaBlock(self.cfg, name="mixer")

        return mixer(norm(xs)) + xs


class MambaModel(nn.Module):
    cfg: Config

    @nn.compact
    def __call__(self, xs):
        embedding = nn.Embed(
            self.cfg.vocab_size,
            self.cfg.d_model,
            dtype=DTYPE_INNER,
            embedding_init=embedding_init,
            name="embedding",
        )
        norm_f = RMSNorm(name="norm_f")
        layers = [
            ResidualBlock(self.cfg, name=f"layers.{i}") for i in range(self.cfg.n_layer)
        ]

        xs = embedding(xs)
        for layer in layers:
            xs = layer(xs)
        xs = norm_f(xs)
        xs = embedding.attend(xs)
        return xs
