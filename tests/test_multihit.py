import netam.multihit as multihit
import netam.framework as framework
import pytest
import pandas as pd
import torch

burrito_params = {
    "batch_size": 1024,
    "learning_rate": 0.1,
    "min_learning_rate": 1e-4,
}


@pytest.fixture
def mini_multihit_train_val_datasets():
    df = pd.read_csv("data/wyatt-10x-1p5m_pcp_2023-11-30_NI.first100.csv.gz")
    crepe = framework.load_crepe("data/cnn_joi_sml-shmoof_small")
    df = multihit.prepare_pcp_df(df, crepe, 500)
    return multihit.train_test_datasets_of_pcp_df(df)


@pytest.fixture
def hitclass_burrito(mini_multihit_train_val_datasets):
    train_data, val_data = mini_multihit_train_val_datasets
    return multihit.MultihitBurrito(
        train_data, val_data, multihit.HitClassModel(), **burrito_params
    )


def test_train(hitclass_burrito):
    before_values = hitclass_burrito.model.values.clone()
    hitclass_burrito.joint_train(epochs=2)
    assert not torch.allclose(hitclass_burrito.model.values, before_values)


def test_serialize(hitclass_burrito):
    hitclass_burrito.save_crepe("test_multihit_crepe")
    new_crepe = framework.load_crepe("test_multihit_crepe")
    assert torch.allclose(new_crepe.model.values, hitclass_burrito.model.values)
