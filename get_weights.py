import argparse
import torch
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--ckpt-file", type=str, default="pytorch_model.bin")
parser.add_argument("--out", type=str, default="mamba-130m.bin")
args = parser.parse_args()

ckpt_file = hf_hub_download(repo_id=args.repo, filename=args.ckpt_file)
ckpt = torch.load(ckpt_file)
# for k, v in ckpt.items(): print(k, v.shape)

n_layers = 0
for k in ckpt:
    if k.startswith("backbone.layers."):
        layer_id = int(k.split(".")[2])
        n_layers = max(n_layers, layer_id + 1)
print(f"n_layers: {n_layers}")

VOCAB_SIZE = 50277
N_LAYER = 24
D_MODEL = 768
D_INNER = D_MODEL * 2
D_CONV = 4
D_STATE = 16
DT_RANK = (D_MODEL + 15) // 16

if n_layers != N_LAYER:
    raise ValueError("unexpected number of layers in file, {n_layers} <> {N_LAYER}")

with open(args.out, "wb") as fobj:
    def write_buffer(w: torch.Tensor, cast_to_float: bool = True):
        assert isinstance(w, torch.Tensor)
        t = w.contiguous().view(-1).detach().cpu()
        if cast_to_float:
            t = t.type(torch.float32)
        t = t.numpy()
        fobj.write(memoryview(t))

    # Let's hope that everything will be properly aligned...
    write_buffer(ckpt["backbone.embedding.weight"])
    for layer_id in range(n_layers):
        prefix = f"backbone.layers.{layer_id}"
        # norm (D_MODEL,)
        write_buffer(ckpt[f"{prefix}.norm.weight"])
        # in_proj{1,2} (D_MODEL, D_INNER).T * 2
        in_projs = ckpt[f"{prefix}.mixer.in_proj.weight"].chunk(2, dim=0)
        write_buffer(in_projs[0])
        write_buffer(in_projs[1])
        # x_proj{1,2,3} (D_INNER, DT_RANK).T (D_INNER, D_STATE).T * 2
        x_projs = ckpt[f"{prefix}.mixer.x_proj.weight"]
        if x_projs.shape != (DT_RANK + 2 * D_STATE, D_INNER):
            raise ValueError(f"unexpected shape for {prefix}.mixer.x_proj.weight {x_projs.shape}")
        write_buffer(x_projs[:DT_RANK])
        write_buffer(x_projs[DT_RANK:DT_RANK + D_STATE])
        write_buffer(x_projs[DT_RANK + D_STATE:])
        # dt_proj (DT_RANK, D_INNER).T
        write_buffer(ckpt[f"{prefix}.mixer.dt_proj.weight"])
        # dt_proj_bias (D_INNER,)
        write_buffer(ckpt[f"{prefix}.mixer.dt_proj.bias"])
        # out_proj (D_INNER, D_MODEL).T
        write_buffer(ckpt[f"{prefix}.mixer.out_proj.weight"])
        # a (D_INNER, D_STATE) exp().neg()
        write_buffer(ckpt[f"{prefix}.mixer.A_log"].exp().neg())
        # d (D_INNER,)
        write_buffer(ckpt[f"{prefix}.mixer.D"])
        # conv1d_weight (D_CONV, D_INNER)
        write_buffer(ckpt[f"{prefix}.mixer.conv1d.weight"].squeeze(1).T)
        # conv1d_bias (D_INNER,)
        write_buffer(ckpt[f"{prefix}.mixer.conv1d.bias"])
    write_buffer(ckpt["backbone.norm_f.weight"])
    write_buffer(ckpt["lm_head.weight"])


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
