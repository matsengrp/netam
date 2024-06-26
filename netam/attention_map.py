"""
We are going to get the attention weights using the [MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) module in PyTorch. These weights are

$$
\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right).
$$

So this tells us that the rows of the attention map correspond to the queries, whereas the columns correspond to the keys.

In our terminology, an attention map is the attention map for a single head. An
"attention maps" object is a collection of attention maps, a tensor where the
first dimension is the number of heads. An "attention mapss" is a list of
attention maps objects, one for each sequence in the batch. An "attention
profile" is some 1-D summary of an attention map, such as the maximum attention
score for each key position. 

# Adapted from https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
"""

import copy
from netam.common import aa_idx_tensor_of_str_ambig, aa_mask_tensor_of


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


def attention_mapss_of(model, which_layer, sequences):
    """
    Get a list of attention maps (across sequences) for the specified layer of
    the model.
    """
    model = copy.deepcopy(model)
    save_output = SaveOutput()
    patch_attention(model.encoder.layers[which_layer].self_attn)
    hook_handle = model.encoder.layers[which_layer].self_attn.register_forward_hook(
        save_output
    )
    for sequence in sequences:
        sequence_idxs = aa_idx_tensor_of_str_ambig(sequence)
        mask = aa_mask_tensor_of(sequence)
        model(sequence_idxs.unsqueeze(0), mask.unsqueeze(0))
    return [out[0].detach().numpy() for out in save_output.outputs]


def attention_profiles_of(model, which_layer, sequences, by):
    """
    Take the mean attention map by heads, then take the maximum attention
    score to get a profile indexed by `by`.

    If by="key", this will return the maximum attention score for each key position.
    If by="query", this will return the maximum attention score for each query position.
    """
    by_to_index_dict = {"query": 1, "key": 0}
    assert by in by_to_index_dict, f"by must be one of {by_to_index_dict.keys()}"
    axis = by_to_index_dict[by]
    attention_mapss = attention_mapss_of(model, which_layer, sequences)
    return [
        attention_maps.mean(axis=0).max(axis=axis) for attention_maps in attention_mapss
    ]
