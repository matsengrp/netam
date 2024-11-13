import netam.pretrained as pretrained
from pathlib import Path
import shutil


def test_names_unique():
    # Check all defined models can be loaded and they have unique names
    assert len(set(pretrained.MODEL_TO_LOCAL.keys())) == sum(
        len(models) for _, _, _, models in pretrained.PACKAGE_LOCATIONS_AND_CONTENTS
    )


def test_load_all_models():
    # Remove cached models:
    shutil.rmtree(Path(pretrained.PRETRAINED_DIR))
    # Check each can be loaded without caching
    for model_name in pretrained.MODEL_TO_LOCAL:
        pretrained.load(model_name)

    # Check each can be loaded with caching
    for model_name in pretrained.MODEL_TO_LOCAL:
        pretrained.load(model_name)
