r"""
We are going to get the attention weights using the [MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) module in PyTorch. These weights are

$$
\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right).
$$

So this tells us that the rows of the attention map correspond to the queries, whereas the columns correspond to the keys.

In our terminology, an attention map is the attention map for a single head. An
"attention maps" object is a collection of attention maps: a tensor where the
first dimension is the number of heads. This assumes all layers have the same
number of heads. An "attention mapss" is a list of attention maps objects, one
for each sequence in the batch. An "attention profile" is some 1-D summary of an
attention map, such as the maximum attention score for each key position.

# Adapted from https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91
"""

import copy

import torch

from netam.sequences import aa_idx_tensor_of_str_ambig, aa_mask_tensor_of


def reshape_tensor(tensor, head_count):
    """Reshape the tensor to include the head dimension.

    Assumes batch size is 1 and squeezes it out.
    """
    assert tensor.size(0) == 1, "Batch size should be 1"
    seq_len, embed_dim = tensor.size(1), tensor.size(2)
    head_dim = embed_dim // head_count
    return tensor.view(seq_len, head_count, head_dim).transpose(0, 1)


class SaveAttentionInfo:
    def __init__(self, head_count):
        self.head_count = head_count
        self.attention_maps = []

    def __call__(self, module, module_in, module_out):
        self.attention_maps.append(module_out[1].clone().squeeze(0))

    def clear(self):
        self.attention_maps = []


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


def attention_mapss_of(model, sequences):
    """Get a list of attention maps (across sequences) as described in the module
    docstring."""
    model = copy.deepcopy(model)
    model.eval()
    layer_count = len(model.encoder.layers)
    # The below assumes all layers have the same number of heads.
    head_count = model.encoder.layers[0].self_attn.num_heads
    save_info = [SaveAttentionInfo(head_count) for _ in range(layer_count)]
    for which_layer, layer in enumerate(model.encoder.layers):
        patch_attention(layer.self_attn)
        layer.self_attn.register_forward_hook(save_info[which_layer])

    for sequence in sequences:
        sequence_idxs = aa_idx_tensor_of_str_ambig(sequence)
        mask = aa_mask_tensor_of(sequence)
        model(sequence_idxs.unsqueeze(0), mask.unsqueeze(0))

    attention_maps = []

    for seq_idx in range(len(sequences)):
        attention_maps.append(
            torch.stack([save.attention_maps[seq_idx] for save in save_info], dim=0)
        )

    return [amap.detach().numpy() for amap in attention_maps]
