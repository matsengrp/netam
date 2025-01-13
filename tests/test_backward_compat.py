import os

import torch
import pytest

from netam.common import BIG, force_spawn
from netam.framework import (
    crepe_exists,
    load_crepe,
    load_pcp_df,
)
from netam.sequences import MAX_AA_TOKEN_IDX, translate_sequence
from netam.models import TransformerBinarySelectionModelWiggleAct
from netam.dasm import (
    DASMBurrito,
    DASMDataset,
    zap_predictions_along_diagonal,
)



# @pytest.fixture(scope="module")
# def dasm_old_burrito(pcp_df):
#     force_spawn()
#     """Fixture that returns the DNSM Burrito object."""
#     pcp_df["in_train"] = True
#     pcp_df.loc[pcp_df.index[-15:], "in_train"] = False
#     train_dataset, val_dataset = DASMDataset.train_val_datasets_of_pcp_df(pcp_df)

#     model = load_crepe("old_models/dasm_13kv1jaffe+v1tang-joint")

#     burrito = DASMBurrito(
#         train_dataset,
#         val_dataset,
#         model,
#         batch_size=32,
#         learning_rate=0.001,
#         min_learning_rate=0.0001,
#     )
#     burrito.joint_train(
#         epochs=1, cycle_count=2, training_method="full", optimize_bl_first_cycle=False
#     )
#     return burrito

def test_old_model_outputs(pcp_df):
    example_seq = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSSGMHWVRQAPGKGLEWVAVIWYDGSNKYYADSVKGRFTISRDNSKNTVYLQMNSLRAEDTAVYYCAREGHSNYPYYYYYMDVWGKGTTVTVSS"
    dasm_crepe = load_crepe("tests/old_models/dasm_13k-v1jaffe+v1tang-joint")
    dnsm_crepe = load_crepe("tests/old_models/dnsm_13k-v1jaffe+v1tang-joint")

    unmodified_df = load_pcp_df(
        "data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz",
    )
    # evaluate on first sequence in pcp_df:
    sequence = unmodified_df["parent"].iloc[0]
    aa_sequence = translate_sequence(sequence)
    print(aa_sequence)
    print(dasm_crepe([example_seq]))
    assert False


    # aa_parents_idxs = pcp_df["aa_parents_idxs"].iloc[0]

    # Test applying models to raw sequences (keep in mind that pcp_df may have
    # tokens added, such as separator tokens, to sequences), and to sequences that have been
    # processed by the Dataset initializer.
