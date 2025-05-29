import torch
import pytest

from netam.pretrained import load_multihit
from netam.common import force_spawn
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.ddsm import (
    DDSMBurrito,
    DDSMDataset,
)
from netam.sequences import (
    MAX_KNOWN_TOKEN_COUNT,
    TOKEN_STR_SORTED,
)
from conftest import get_pcp_df

torch.set_printoptions(precision=10)


@pytest.fixture(scope="module", params=["heavy", "paired"])
def ddsm_burrito(request):
    pcp_df = get_pcp_df(request.param)
    force_spawn()
    """Fixture that returns the DNSM Burrito object."""
    pcp_df["in_train"] = True
    pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
    model = TransformerBinarySelectionModelWiggleAct(
        nhead=2,
        d_model_per_head=4,
        dim_feedforward=256,
        layer_count=2,
        output_dim=20,
        model_type="ddsm",
    )

    train_dataset, val_dataset = DDSMDataset.train_val_datasets_of_pcp_df(
        pcp_df,
        MAX_KNOWN_TOKEN_COUNT,
        multihit_model=load_multihit(model.multihit_model_name),
    )

    burrito = DDSMBurrito(
        train_dataset,
        val_dataset,
        model,
        batch_size=32,
        learning_rate=0.001,
        min_learning_rate=0.0001,
    )
    burrito.joint_train(
        epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
    )
    return burrito


def test_build_selection_matrix_from_parent(ddsm_burrito):
    parent = ddsm_burrito.val_dataset.nt_parents[0]
    parent_aa_idxs = ddsm_burrito.val_dataset.aa_parents_idxss[0]
    aa_mask = ddsm_burrito.val_dataset.masks[0]
    aa_parent = "".join(TOKEN_STR_SORTED[i] for i in parent_aa_idxs)
    # This won't work if we start testing with ambiguous sequences
    aa_parent = aa_parent.replace("X", "")

    separator_idx = aa_parent.index("^") * 3
    light_chain_seq = parent[:separator_idx]
    heavy_chain_seq = parent[separator_idx + 3 :]

    direct_val = ddsm_burrito.build_selection_matrix_from_parent_aa(
        parent_aa_idxs, aa_mask
    )

    indirect_val = ddsm_burrito._build_selection_matrix_from_parent(
        (light_chain_seq, heavy_chain_seq)
    )

    assert torch.allclose(direct_val[: len(indirect_val[0])], indirect_val[0])
    assert torch.allclose(
        direct_val[len(indirect_val[0]) + 1 :][: len(indirect_val[1])], indirect_val[1]
    )
