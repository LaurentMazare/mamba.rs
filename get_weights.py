import argparse
import json
import torch
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--which", type=str, default="130m")
parser.add_argument("--out", type=str)
args = parser.parse_args()

SUPPORTED_W = ("130m", "370m", "790m", "1.4b", "2.8b")
if args.which not in SUPPORTED_W:
    raise ValueError(f"the which argument should be in {SUPPORTED_W}")
repo_id = f"state-spaces/mamba-{args.which}"
print(f"getting weights and config from {repo_id}")

config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
with open(config_file, "r") as f_obj:
    config = json.load(f_obj)
print(config)

ckpt_file = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
ckpt = torch.load(ckpt_file)
def get_tensor(name: str, exp_shape):
    tensor = ckpt[name]
    if tensor.shape != exp_shape:
        raise ValueError(f"unexpected shape for {name}, {tensor.shape} <> {exp_shape}")
    return tensor
# for k, v in ckpt.items(): print(k, v.shape)

n_layers = 0
for k in ckpt:
    if k.startswith("backbone.layers."):
        layer_id = int(k.split(".")[2])
        n_layers = max(n_layers, layer_id + 1)
print(f"n_layers: {n_layers}")

VOCAB_SIZE = 50280 # padded to a multiple of 8, should be 50277 otherwise
N_LAYER = config["n_layer"]
D_MODEL = config["d_model"]
D_INNER = D_MODEL * 2
D_CONV = 4
D_STATE = 16
DT_RANK = (D_MODEL + 15) // 16

if n_layers != N_LAYER:
    raise ValueError(f"unexpected number of layers in file, {n_layers} <> {N_LAYER}")

out_file = args.out
if out_file is None:
    out_file = f"mamba-{args.which}.bin"
print(f"writing model file {out_file}")
with open(out_file, "wb") as fobj:
    def write_buffer(w: torch.Tensor, cast_to_float: bool = True):
        assert isinstance(w, torch.Tensor)
        t = w.contiguous().view(-1).detach().cpu()
        if cast_to_float:
            t = t.type(torch.float32)
        t = t.numpy()
        fobj.write(memoryview(t))

    # Let's hope that everything will be properly aligned...
    t = get_tensor("backbone.embedding.weight", (VOCAB_SIZE, D_MODEL))
    write_buffer(t)
    for layer_id in range(n_layers):
        prefix = f"backbone.layers.{layer_id}"
        # norm (D_MODEL,)
        t = get_tensor(f"{prefix}.norm.weight", (D_MODEL,))
        write_buffer(t)
        # in_proj{1,2} (D_MODEL, D_INNER).T * 2
        in_projs = get_tensor(f"{prefix}.mixer.in_proj.weight", (D_INNER * 2, D_MODEL))
        in_projs = in_projs.chunk(2, dim=0)
        write_buffer(in_projs[0])
        write_buffer(in_projs[1])
        # x_proj{1,2,3} (D_INNER, DT_RANK).T (D_INNER, D_STATE).T * 2
        x_projs = get_tensor(f"{prefix}.mixer.x_proj.weight", (DT_RANK + 2*D_STATE, D_INNER))
        write_buffer(x_projs[:DT_RANK])
        write_buffer(x_projs[DT_RANK:DT_RANK + D_STATE])
        write_buffer(x_projs[DT_RANK + D_STATE:])
        # dt_proj (DT_RANK, D_INNER).T
        t = get_tensor(f"{prefix}.mixer.dt_proj.weight", (D_INNER, DT_RANK))
        write_buffer(t)
        # dt_proj_bias (D_INNER,)
        t = get_tensor(f"{prefix}.mixer.dt_proj.bias", (D_INNER,))
        write_buffer(t)
        # out_proj (D_INNER, D_MODEL).T
        t = get_tensor(f"{prefix}.mixer.out_proj.weight", (D_MODEL, D_INNER))
        write_buffer(t)
        # a (D_INNER, D_STATE) exp().neg()
        t = get_tensor(f"{prefix}.mixer.A_log", (D_INNER, D_STATE))
        write_buffer(t.exp().neg())
        # d (D_INNER,)
        t = get_tensor(f"{prefix}.mixer.D", (D_INNER,))
        write_buffer(t)
        # conv1d_weight (D_CONV, D_INNER)
        t = get_tensor(f"{prefix}.mixer.conv1d.weight", (D_INNER, 1, D_CONV))
        write_buffer(t.squeeze(1).T)
        # conv1d_bias (D_INNER,)
        t = get_tensor(f"{prefix}.mixer.conv1d.bias", (D_INNER,))
        write_buffer(t)
    t = get_tensor("backbone.norm_f.weight", (D_MODEL,))
    write_buffer(t)
    t = get_tensor("lm_head.weight", (VOCAB_SIZE, D_MODEL))
    write_buffer(t)


# Typical layout
# backbone.embedding.weight torch.Size([50280, 768])
# backbone.layers.0.mixer.D torch.Size([1536])
# backbone.layers.0.mixer.in_proj.weight torch.Size([3072, 768])
# backbone.layers.0.mixer.conv1d.weight torch.Size([1536, 1, 4])
# backbone.layers.0.mixer.conv1d.bias torch.Size([1536])
# backbone.layers.0.mixer.x_proj.weight torch.Size([80, 1536])
# backbone.layers.0.mixer.dt_proj.weight torch.Size([1536, 48])
# backbone.layers.0.mixer.dt_proj.bias torch.Size([1536])
# backbone.layers.0.mixer.A_log torch.Size([1536, 16])
# backbone.layers.0.mixer.out_proj.weight torch.Size([768, 1536])
# backbone.layers.0.norm.weight torch.Size([768])
# backbone.norm_f.weight torch.Size([768])
# lm_head.weight torch.Size([50280, 768])
