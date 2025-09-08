import pytest

from netam.sequences import MAX_KNOWN_TOKEN_COUNT
from netam.common import force_spawn
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.dnsm import DNSMBurrito, DNSMDataset
from netam.framework import (
    load_pcp_df,
    add_shm_model_outputs_to_pcp_df,
)
from netam import pretrained
import random


# Function to randomly insert 'N' in sequences
def randomize_with_ns(parent_seq, child_seq, avoid_masked_equality=True):
    old_parent = parent_seq
    old_child = child_seq
    seq_length = len(parent_seq)
    try:
        first_mut = next(
            (idx, p, c)
            for idx, (p, c) in enumerate(zip(parent_seq, child_seq))
            if p != c
        )
    except StopIteration:
        return parent_seq, child_seq

    # Decide which type of modification to apply
    modification_type = random.choice(["same_site", "different_site", "chunk", "none"])

    if modification_type == "same_site":
        # Select random positions to place 'N' in both parent and child at the same positions
        num_ns = random.randint(
            1, seq_length // 3
        )  # Random number of Ns, max third the sequence length
        positions = random.sample(range(seq_length), num_ns)
        parent_seq = "".join(
            ["N" if i in positions else base for i, base in enumerate(parent_seq)]
        )
        child_seq = "".join(
            ["N" if i in positions else base for i, base in enumerate(child_seq)]
        )

    elif modification_type == "different_site":
        # Insert 'N's at random positions in parent and child, but not the same positions
        num_ns_parent = random.randint(1, seq_length // 3)
        num_ns_child = random.randint(1, seq_length // 3)
        positions_parent = random.sample(range(seq_length), num_ns_parent)
        positions_child = random.sample(range(seq_length), num_ns_child)

        parent_seq = "".join(
            [
                "N" if i in positions_parent else base
                for i, base in enumerate(parent_seq)
            ]
        )
        child_seq = "".join(
            ["N" if i in positions_child else base for i, base in enumerate(child_seq)]
        )

    elif modification_type == "chunk":
        # Replace a chunk of bases with 'N's in both parent and child
        chunk_size = random.randint(1, seq_length // 3)
        start_pos = random.randint(0, seq_length - chunk_size)
        parent_seq = (
            parent_seq[:start_pos]
            + "N" * chunk_size
            + parent_seq[start_pos + chunk_size :]
        )
        child_seq = (
            child_seq[:start_pos]
            + "N" * chunk_size
            + child_seq[start_pos + chunk_size :]
        )

    if parent_seq == child_seq:
        # If sequences are the same, put one mutated site back in:
        idx, p, c = first_mut
        parent_seq = parent_seq[:idx] + p + parent_seq[idx + 1 :]
        child_seq = child_seq[:idx] + c + child_seq[idx + 1 :]
    if avoid_masked_equality:
        codon_pairs = [
            (parent_seq[i * 3 : (i + 1) * 3], child_seq[i * 3 : (i + 1) * 3])
            for i in range(seq_length // 3)
        ]
        if all(
            p == c
            for p, c in filter(
                lambda pair: "N" not in pair[0] and "N" not in pair[1], codon_pairs
            )
        ):
            # put original codon containing a mutation back in.
            idx, p, c = first_mut
            codon_start = (idx // 3) * 3
            codon_end = codon_start + 3
            parent_seq = (
                parent_seq[:codon_start]
                + old_parent[codon_start:codon_end]
                + parent_seq[codon_end:]
            )
            child_seq = (
                child_seq[:codon_start]
                + old_child[codon_start:codon_end]
                + child_seq[codon_end:]
            )

    assert len(parent_seq) == len(child_seq)
    assert len(parent_seq) == seq_length

    return parent_seq, child_seq


@pytest.fixture
def ambig_pcp_df():
    random.seed(1)
    df = load_pcp_df(
        "data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz",
    )
    # Apply the random N adding function to each row
    df[["parent_heavy", "child_heavy"]] = df.apply(
        lambda row: tuple(
            randomize_with_ns(row["parent_heavy"][:-3], row["child_heavy"][:-3])
        ),
        axis=1,
        result_type="expand",
    )

    df = add_shm_model_outputs_to_pcp_df(
        df,
        pretrained.load("ThriftyHumV0.2-59"),
    )
    return df


@pytest.fixture
def dnsm_model():
    return TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        model_type="dnsm",
    )


def test_dnsm_burrito(ambig_pcp_df, dnsm_model):
    """Fixture that returns the DNSM Burrito object."""
    force_spawn()
    ambig_pcp_df["in_train"] = True
    ambig_pcp_df.loc[ambig_pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DNSMDataset.train_val_datasets_of_pcp_df(
        ambig_pcp_df,
        MAX_KNOWN_TOKEN_COUNT,
        multihit_model=pretrained.load_multihit(
            dnsm_model.multihit_model_name, device=None
        ),
    )

    burrito = DNSMBurrito(
        train_dataset,
        val_dataset,
        dnsm_model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.joint_train(epochs=1, cycle_count=2, training_method="full")


@pytest.fixture
def dasm_model():
    return TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
        model_type="dasm",
    )


def test_dasm_burrito(ambig_pcp_df, dasm_model):
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    ambig_pcp_df["in_train"] = True
    ambig_pcp_df.loc[ambig_pcp_df.index[-15:], "in_train"] = False
    train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(
        ambig_pcp_df,
        MAX_KNOWN_TOKEN_COUNT,
        multihit_model=pretrained.load_multihit(
            dasm_model.multihit_model_name, device=None
        ),
    )

    burrito = DASMBurrito(
        train_dataset,
        val_dataset,
        dasm_model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.joint_train(
        epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
    )
