# mamba.rs

Like [llama2.rs](https://github.com/srush/llama2.rs) (by @srush) but for
[mamba](https://arxiv.org/abs/2312.00752): this is a small standalone
implementation of mamba, for inference only on CPU.

```bash
# Download the tokenizer config.
wget https://huggingface.co/EleutherAI/gpt-neox-20b/raw/main/tokenizer.json

# Generate the weight files.
python get_weights.py
```
