"""This module provides a simple interface for downloading pre-trained models.

It was inspired by the `load_model` module of [AbLang2](https://github.com/oxpig/AbLang2).
"""

import os
import zipfile
import pkg_resources

import requests

from netam.framework import load_crepe

# This throws a deprecation warning. It could also be done by looking at
# __file__, or by using importlib.resources.
PRETRAINED_DIR = pkg_resources.resource_filename(__name__, "_pretrained")

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
)

LOCAL_TO_REMOTE = {}
MODEL_TO_LOCAL = {}
LOCAL_TO_DIR = {}

for local_file, remote, models_dir, models in PACKAGE_LOCATIONS_AND_CONTENTS:
    LOCAL_TO_REMOTE[local_file] = remote

    for model in models:
        MODEL_TO_LOCAL[model] = (local_file, models_dir)


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

    local_crepe_path = local_path_for_model(model_name)
    return load_crepe(local_crepe_path, device=device)
