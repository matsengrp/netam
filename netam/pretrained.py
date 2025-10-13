"""This module provides a simple interface for downloading pre-trained models.

It was inspired by the `load_model` module of [AbLang2](https://github.com/oxpig/AbLang2).
"""

import os
import zipfile
from importlib.resources import files
import numpy as np

import requests

from netam.framework import load_crepe
from netam.models import HitClassModel


pretrained_path = files(__package__).joinpath("_pretrained")
PRETRAINED_DIR = str(pretrained_path)

PACKAGE_LOCATIONS_AND_CONTENTS = (
    # Order of entries:
    # * Local file name
    # * Remote URL
    # * Directory in which the models appear after extraction (must match path determined by archive)
    # * List of models in the package
    [
        "thrifty-0.2.0.zip",
        "https://github.com/matsengrp/thrifty-models/archive/refs/tags/v0.2.0.zip",
        "thrifty-models-0.2.0/models",
        [
            "ThriftyHumV0.2-20",
            "ThriftyHumV0.2-45",
            "ThriftyHumV0.2-59",
        ],
    ],
    [
        "dnsm-1.0.0.zip",
        "https://github.com/matsengrp/dnsm-models/archive/v1.0.0.zip",
        "dnsm-models-1.0.0/models",
        [
            "DNSMHumV1.0-1M",
            "DNSMHumV1.0-4M",
        ],
    ],
    [
        "dasm-1.0.0.zip",
        "https://github.com/matsengrp/dasm-models/archive/v1.0.0.zip",
        "dasm-models-1.0.0/models",
        [
            "DASMHumV1.0-4M",
        ],
    ],
)

LOCAL_TO_REMOTE = {}
MODEL_TO_LOCAL = {}
LOCAL_TO_DIR = {}

for local_file, remote, models_dir, models in PACKAGE_LOCATIONS_AND_CONTENTS:
    LOCAL_TO_REMOTE[local_file] = remote

    for model in models:
        MODEL_TO_LOCAL[model] = (local_file, models_dir)


# Names here are arbitrarily chosen (they are not processed or interpreted in
# any way by the code), and should describe the origin of the
# multihit model.
PRETRAINED_MULTIHIT_MODELS = {
    # Trained using the notebook
    # thrifty-experiments-1/human/multihit_model_exploration.ipynb
    "ThriftyHumV0.2-59-hc-tangshm": (-0.1626, 0.0692, 0.5076),
    "ThriftyHumV0.2-59-hc-shmoof": (-0.2068, 0.1603, 0.6317),
}


def local_path_for_model(model_name: str):
    """Return the local path for a model, downloading it if necessary."""

    if model_name not in MODEL_TO_LOCAL:
        raise ValueError(f"Model {model_name} not found in pre-trained models.")

    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    local_package, models_dir = MODEL_TO_LOCAL[model_name]
    local_package_path = os.path.join(PRETRAINED_DIR, local_package)

    if not os.path.exists(local_package_path):
        url = LOCAL_TO_REMOTE[local_package]
        print(f"Fetching models: downloading {url} to {local_package_path}")
        response = requests.get(url)
        response.raise_for_status()
        with open(local_package_path, "wb") as f:
            f.write(response.content)
        if local_package.endswith(".zip"):
            with zipfile.ZipFile(local_package_path, "r") as zip_ref:
                zip_ref.extractall(path=PRETRAINED_DIR)
        else:
            raise ValueError(f"Unknown file type for {local_package}")
    else:
        print(f"Using cached models: {local_package_path}")

    local_crepe_path = os.path.join(PRETRAINED_DIR, models_dir, model_name)

    if not os.path.exists(local_crepe_path + ".yml"):
        raise ValueError(f"Model {local_crepe_path} not found in pre-trained models.")
    if not os.path.exists(local_crepe_path + ".pth"):
        raise ValueError(f"Model {model_name} missing model weights.")

    return local_crepe_path


def load(model_name: str, device=None):
    """Load a pre-trained model.

    If the model is not already downloaded, it will be downloaded from the appropriate
    URL and stashed in the PRETRAINED_DIR.
    """
    print(f"Loading model {model_name}")
    local_crepe_path = local_path_for_model(model_name)
    return load_crepe(local_crepe_path, device=device)


def load_multihit(model_name: str, device=None):
    """Load a pre-trained multihit model."""
    if model_name is None:
        return None
    else:
        try:
            parameters = PRETRAINED_MULTIHIT_MODELS[model_name]
        except KeyError:
            raise ValueError(
                f"Model {model_name} not found in pre-trained multihit models."
            )
        print(f"Loading multihit model {model_name}")
        model = HitClassModel.from_weights(parameters)
        return model.to(device)


def name_and_multihit_model_match(model_name: str, multihit_model: HitClassModel):
    """Check if the model name and multihit model match."""
    if model_name is None:
        return multihit_model is None
    else:
        if multihit_model is None:
            return False
        elif model_name in PRETRAINED_MULTIHIT_MODELS:
            return np.allclose(
                multihit_model.to_weights(), PRETRAINED_MULTIHIT_MODELS[model_name]
            )
        else:
            return model_name == str(multihit_model.to_weights())
