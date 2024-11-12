import pytest
from netam.framework import (
    load_pcp_df,
    add_shm_model_outputs_to_pcp_df,
)
from netam import pretrained


@pytest.fixture(scope="module")
def pcp_df():
    df = load_pcp_df(
        "data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz",
    )
    df = add_shm_model_outputs_to_pcp_df(
        df,
        pretrained.load("ThriftyHumV1.0-45"),
    )
    return df
