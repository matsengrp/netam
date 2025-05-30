import pytest
from netam.framework import (
    load_pcp_df,
    add_shm_model_outputs_to_pcp_df,
)
from netam import pretrained

data_files = {
    "heavy": "data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz",
    "paired": "data/wyatt-10x-1p5m_paired-merged_fs-all_pcp_2024-11-21_no-naive_sample100_DXSM_Valid.csv.gz",
}


def get_pcp_df(name):
    df = load_pcp_df(data_files[name])
    df = add_shm_model_outputs_to_pcp_df(
        df,
        pretrained.load("ThriftyHumV0.2-59"),
    )
    return df


@pytest.fixture(scope="module")
def pcp_df():
    return get_pcp_df("heavy")


@pytest.fixture(scope="module")
def pcp_df_paired():
    return get_pcp_df("paired")
