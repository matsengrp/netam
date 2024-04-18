import math
import inspect
import itertools

import numpy as np
import torch
import torch.optim as optim
from torch import nn, Tensor

BIG = 1e9
SMALL_PROB = 1e-6
BASES = ["A", "C", "G", "T"]
BASES_AND_N_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
AA_STR_SORTED = "ACDEFGHIKLMNPQRSTVWY"
AA_STR_SORTED_AMBIG = AA_STR_SORTED + "X"
MAX_AMBIG_AA_IDX = len(AA_STR_SORTED_AMBIG) - 1

# https://www.ncbi.nlm.nih.gov/nuccore/GU980702.1
VRC01_NT_SEQ = (
    "CAGGTGCAGCTGGTGCAGTCTGGGGGTCAGATGAAGAAGCCTGGCGAGTCGATGAGAATT"
    "TCTTGTCGGGCTTCTGGATATGAATTTATTGATTGTACGCTAAATTGGATTCGTCTGGCC"
    "CCCGGAAAAAGGCCTGAGTGGATGGGATGGCTGAAGCCTCGGGGGGGGGCCGTCAACTAC"
    "GCACGTCCACTTCAGGGCAGAGTGACCATGACTCGAGACGTTTATTCCGACACAGCCTTT"
    "TTGGAGCTGCGCTCGTTGACAGTAGACGACACGGCCGTCTACTTTTGTACTAGGGGAAAA"
    "AACTGTGATTACAATTGGGACTTCGAACACTGGGGCCGGGGCACCCCGGTCATCGTCTCA"
    "TCA"
)


def generate_kmers(kmer_length):
    # Our strategy for kmers is to have a single representation for any kmer that isn't in ACGT.
    # This is the first one, which is simply "N", and so this placeholder value is 0.
    all_kmers = ["N"] + [
        "".join(p) for p in itertools.product(BASES, repeat=kmer_length)
    ]
    assert len(all_kmers) < torch.iinfo(torch.int32).max
    return all_kmers


def kmer_to_index_of(all_kmers):
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


def aa_idx_tensor_of_str_ambig(aa_str):
    """Return the indices of the amino acids in a string, allowing the ambiguous character."""
    try:
        return torch.tensor(
            [AA_STR_SORTED_AMBIG.index(aa) for aa in aa_str], dtype=torch.int
        )
    except ValueError:
        print(f"Found an invalid amino acid in the string: {aa_str}")
        raise


def generic_mask_tensor_of(ambig_symb, seq_str, length=None):
    """Return a mask tensor indicating non-empty and non-ambiguous sites. Sites
    beyond the length of the sequence are masked."""
    if length is None:
        length = len(seq_str)
    mask = torch.zeros(length, dtype=torch.bool)
    if len(seq_str) < length:
        seq_str += ambig_symb * (length - len(seq_str))
    else:
        seq_str = seq_str[:length]
    mask[[c != ambig_symb for c in seq_str]] = 1
    return mask


def nt_mask_tensor_of(*args, **kwargs):
    return generic_mask_tensor_of("N", *args, **kwargs)


def aa_mask_tensor_of(*args, **kwargs):
    return generic_mask_tensor_of("X", *args, **kwargs)


def informative_site_count(seq_str):
    return sum(c != "N" for c in seq_str)


def clamp_probability(x: Tensor) -> Tensor:
    return torch.clamp(x, min=SMALL_PROB, max=(1.0 - SMALL_PROB))


def print_parameter_count(model):
    total = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only count parameters in leaf modules
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {num_params} parameters")
            total += num_params
    print("-----")
    print(f"total: {total} parameters")


def parameter_count_of_model(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stack_heterogeneous(tensors, pad_value=0.0):
    """
    Stack an iterable of 1D or 2D torch.Tensor objects of different lengths along the first dimension into a single tensor.

    Parameters:
    tensors (iterable): An iterable of 1D or 2D torch.Tensor objects with variable lengths in the first dimension.
    pad_value (number): The value used for padding shorter tensors. Default is 0.

    Returns:
    torch.Tensor: A stacked tensor with all input tensors padded to the length of the longest tensor in the first dimension.
    """
    if tensors is None or len(tensors) == 0:
        return torch.Tensor()  # Return an empty tensor if no tensors are provided

    dim = tensors[0].dim()
    if dim not in [1, 2]:
        raise ValueError("This function only supports 1D or 2D tensors.")

    max_length = max(tensor.size(0) for tensor in tensors)

    if dim == 1:
        # If 1D, simply pad the end of the tensor.
        padded_tensors = [
            torch.nn.functional.pad(
                tensor, (0, max_length - tensor.size(0)), value=pad_value
            )
            for tensor in tensors
        ]
    else:
        # If 2D, pad the end of the first dimension (rows); the argument to pad
        # is a tuple of (padding_left, padding_right, padding_top,
        # padding_bottom)
        padded_tensors = [
            torch.nn.functional.pad(
                tensor, (0, 0, 0, max_length - tensor.size(0)), value=pad_value
            )
            for tensor in tensors
        ]

    return torch.stack(padded_tensors)


def pick_device():
    # check that CUDA is usable
    def check_CUDA():
        try:
            torch._C._cuda_init()
            return True
        except:
            return False

    if torch.backends.cudnn.is_available() and check_CUDA():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def print_tensor_devices(scope="local"):
    """
    Print the devices of all PyTorch tensors in the given scope.

    Args:
    scope (str): 'local' for local scope, 'global' for global scope.
    """
    if scope == "local":
        frame = inspect.currentframe()
        variables = frame.f_back.f_locals
    elif scope == "global":
        variables = globals()
    else:
        raise ValueError("Scope must be 'local' or 'global'.")

    for var_name, var_value in variables.items():
        if isinstance(var_value, torch.Tensor):
            print(f"{var_name}: {var_value.device}")


# Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # assert that d_model is even
        assert d_model % 2 == 0, "d_model must be even for PositionalEncoding"

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
