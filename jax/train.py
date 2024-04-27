import argparse
import json
import os

import tboard
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import orbax.checkpoint as orbax
import optax
from tokenizers import Tokenizer
from data import load_glob
from model import Config, MambaModel

print(os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument("--bsize", type=int, default=96)
parser.add_argument("--csize", type=int, default=2048)
parser.add_argument(
    "--lr", type=float, default=30e-4
)  # 6e-4 x 3 (see the paper) x sqrt(96/256)
parser.add_argument("--checkpoint-path", type=str, default=None)
parser.add_argument("--glob-data", type=str)
parser.add_argument("--tboard-path", type=str)
parser.add_argument("--total-steps", type=int, default=50000)
args = parser.parse_args()

tb = tboard.EventWriter(args.tboard_path)

cfg130m = Config(scan_op="associative-scan")
print(cfg130m)
model = MambaModel(cfg130m)

local_mesh = jax.sharding.Mesh(jax.local_devices(), ("gpus"))
none_sharding = jax.sharding.NamedSharding(local_mesh, P())
local_sharding = jax.sharding.NamedSharding(local_mesh, P("gpus"))


def prod(vs):
    res = 1
    for v in vs:
        res *= v
    return res


schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.lr,
    warmup_steps=args.total_steps // 20,
    decay_steps=args.total_steps // 2,
    end_value=1e-5,
)


# Do not apply weight decay on A_log or D.
def wd_mask(params):
    mask = jax.tree_util.tree_map_with_path(
        lambda x, y: x[-1].key not in ["A_log", "D"], params
    )
    return mask


# Parameters specified on page 30 of the mamba paper.
optimizer = optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.1, mask=wd_mask)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optimizer,
)
ckpt_manager = None
if args.checkpoint_path:
    if os.path.exists(args.checkpoint_path):
        print(f"Checkpoint already exists: {args.checkpoint_path}.")

    os.makedirs(args.checkpoint_path, exist_ok=True)
    with open(os.path.join(args.checkpoint_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    ckpt_manager = orbax.CheckpointManager(
        args.checkpoint_path,
        {
            "params": orbax.PyTreeCheckpointer(),
            "opt_state": orbax.PyTreeCheckpointer(),
        },
        options=orbax.CheckpointManagerOptions(max_to_keep=2),
    )


def init():
    params = model.init(jax.random.PRNGKey(0), jnp.array([[209] * args.csize] * 1))
    print(
        "total weights",
        sum([prod(k.shape) for k in jax.tree_util.tree_flatten(params)[0]]),
    )
    opt_state = optimizer.init(params)
    return params, opt_state


params, opt_state = jax.jit(
    init,
    out_shardings=(none_sharding, none_sharding),
)()


def loss(params, x, y):
    logits = model.apply(params, x)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(ce)


def step(params, opt_state, x, y):
    value, grads = jax.value_and_grad(loss)(params, x, y)
    update, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, opt_state, value


step = jax.jit(
    step,
    in_shardings=(none_sharding, none_sharding, local_sharding, local_sharding),
    out_shardings=(none_sharding, none_sharding, none_sharding),
)


tokenizer = Tokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

data_iter = load_glob(
    args.glob_data,
    tokenizer,
    args.bsize,
    args.csize,
)
value_avg, nex = 0.0, 0
for batch_idx, batch in enumerate(data_iter):
    batch, _c_bpb = batch
    batch = jnp.asarray(batch).reshape(args.bsize, -1)
    x = batch[:, :-1]
    y = batch[:, 1:]

    params, opt_state, value = step(params, opt_state, x, y)
    value_avg += value.item()
    nex += 1
    if batch_idx == 0 or nex % 100 == 0:
        nll = value_avg / nex
        tb.add_scalar("train-nll", nll, batch_idx)
        print(batch_idx, nll)
        value_avg, nex = 0.0, 0

    if ckpt_manager is not None and batch_idx > 0 and batch_idx % 1000 == 0:
        ckpt_manager.save(
            batch_idx,
            {
                "params": params,
                "opt_state": opt_state,
            },
        )

    if batch_idx >= args.total_steps:
        break
