import glob
import math
import json
import numpy as np


def tokenize(text, tokenizer, bos=True, eos=False):
    nl_piece = tokenizer.encode("\n").ids[-1]
    # Hardcode the bos token to 0 (ok for the GPT2 tokenizer)
    tokens = [0] if bos else []
    for line_idx, line in enumerate(text.split("\n")):
        if line_idx > 0:
            tokens.append(nl_piece)
        for id in tokenizer.encode(line).ids:
            tokens.append(id)
    if eos:
        raise ValueError("todo")
    return tokens


def detokenize(tokens, tokenizer):
    text = tokenizer.decode(tokens)
    return text.replace("\n ", "\n")


def jsonl_iterator(filename, remove_example=lambda x: False):
    while True:
        fin = open(filename)
        for line in fin:
            data = json.loads(line)
            if not remove_example(data):
                yield data


def token_iterator(iterator, tokenizer, field="text"):
    n_chars, n_tokens = 0, 0
    for data in iterator:
        tokens = tokenize(data[field], tokenizer)
        n_chars += len(data[field])
        n_tokens += len(tokens)
        c_bpb = n_tokens / n_chars / math.log(2)
        yield {"tokens": tokens, "c_bpb": c_bpb}


def batch_iterator(iterator, bsz, csz):
    tokens = []
    n = bsz * (csz + 1)
    for data in iterator:
        tokens += data["tokens"]
        if len(tokens) > n:
            a = len(tokens) // n
            x = np.asarray(tokens[: a * n]).reshape(bsz, a, -1)
            for i in range(a):
                yield x[:, i, :], data["c_bpb"]
            tokens = tokens[a * n :]


def load_data(filename, tokenizer, bsz, csz):
    tokens = []
    n = bsz * (csz + 1)
    fin = open(filename)
    n_chars, n_tokens = 0, 0
    for line in fin:
        data = json.loads(line)
        t = tokenize(data["text"], tokenizer)
        tokens += t
        n_chars += len(data["text"])
        n_tokens += len(t)
        while len(tokens) >= n:
            c_bpb = n_tokens / n_chars / math.log(2)
            yield np.asarray(tokens[:n]).reshape(bsz, csz + 1), c_bpb
            tokens = tokens[n:]


def load_glob(g, tokenizer, bsz, csz):
    while True:
        for filename in glob.glob(g):
            for token in load_data(filename, tokenizer, bsz, csz):
                yield token
