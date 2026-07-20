import tiktoken

base_enc = tiktoken.get_encoding("gpt2")

special_tokens = {
    "<|BOS|>": base_enc.n_vocab,
    "<|USER|>": base_enc.n_vocab + 1,
    "<|ASSISTANT|>": base_enc.n_vocab + 2,
    "<|EOT|>": base_enc.n_vocab + 3,
    "<|PAD|>": base_enc.n_vocab + 4,
    "<|EOS|>": base_enc.n_vocab + 5,
}

enc = tiktoken.Encoding(
    name="gpt2_custom",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens=special_tokens,
)


def num_tokens(data):
    """
    Returns the number of tokens in:
    - a string
    - a list of message strings
    """
    if isinstance(data, str):
        text = data
    elif isinstance(data, list):
        text = "".join(data)
    else:
        raise TypeError("Input must be a string or a list of strings.")

    return len(enc.encode(text, allowed_special="all"))




