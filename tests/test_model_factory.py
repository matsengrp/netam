"""Tests for model factory functions."""

import pytest
import torch
import tempfile
import yaml
import os
from netam.model_factory import (
    create_selection_model_from_dict,
    create_selection_model_from_yaml,
    create_selection_model_from_file,
    create_model_from_preset,
    list_available_models,
    list_available_presets,
    get_model_info,
    describe_model,
    validate_config_schema,
    _load_preset_config,
)


def test_create_selection_model_from_dict_single_value():
    """Test model creation from dictionary configuration - SingleValueBinarySelectionModel."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    assert model.__class__.__name__ == "SingleValueBinarySelectionModel"
    assert model.model_type == "dasm"
    assert model.known_token_count == 21
    assert model.output_dim == 20


def test_create_selection_model_from_dict_parent_independent():
    """Test ParentIndependentBinarySelectionModel creation."""
    config = {
        "model_class": "ParentIndependentBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    assert model.__class__.__name__ == "ParentIndependentBinarySelectionModel"
    assert model.model_type == "dasm"
    assert model.known_token_count == 21
    assert model.output_dim == 20


def test_create_selection_model_from_dict_transformer():
    """Test transformer model creation."""
    config = {
        "model_class": "TransformerBinarySelectionModelWiggleAct",
        "hparams": {
            "model_type": "dasm",
            "known_token_count": 21,
            "output_dim": 20,
            "nhead": 4,
            "d_model_per_head": 4,
            "dim_feedforward": 1024,
            "layer_count": 3,
            "dropout_prob": 0.1,
        },
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    assert model.__class__.__name__ == "TransformerBinarySelectionModelWiggleAct"
    assert model.model_type == "dasm"
    assert model.known_token_count == 21
    assert model.output_dim == 20
    assert model.nhead == 4
    assert model.d_model_per_head == 4
    assert model.dim_feedforward == 1024
    # Access layer_count and dropout_prob through encoder properties
    assert model.encoder.num_layers == 3
    assert model.pos_encoder.dropout.p == 0.1


def test_create_selection_model_from_dict_transformer_lin_act():
    """Test TransformerBinarySelectionModelLinAct creation."""
    config = {
        "model_class": "TransformerBinarySelectionModelLinAct",
        "hparams": {
            "model_type": "dnsm",
            "known_token_count": 21,
            "output_dim": 20,
            "nhead": 2,
            "d_model_per_head": 8,
            "dim_feedforward": 512,
            "layer_count": 2,
        },
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    assert model.__class__.__name__ == "TransformerBinarySelectionModelLinAct"
    assert model.model_type == "dnsm"
    assert model.nhead == 2
    assert model.d_model_per_head == 8


def test_create_selection_model_from_dict_bidirectional():
    """Test BidirectionalTransformerBinarySelectionModel creation."""
    config = {
        "model_class": "BidirectionalTransformerBinarySelectionModel",
        "hparams": {
            "model_type": "dasm",
            "known_token_count": 21,
            "output_dim": 20,
            "nhead": 4,
            "d_model_per_head": 6,
            "dim_feedforward": 256,
            "layer_count": 2,
        },
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    assert model.__class__.__name__ == "BidirectionalTransformerBinarySelectionModel"
    assert model.model_type == "dasm"
    assert model.nhead == 4
    assert model.d_model_per_head == 6


def test_create_selection_model_invalid_class():
    """Test error handling for invalid model class."""
    config = {"model_class": "NonexistentModel", "hparams": {}}

    device = torch.device("cpu")

    with pytest.raises(ValueError) as excinfo:
        create_selection_model_from_dict(config, device)

    assert "Unknown model class: NonexistentModel" in str(excinfo.value)
    assert "Available model classes:" in str(excinfo.value)


def test_create_selection_model_empty_hparams():
    """Test model creation with empty hparams dictionary."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm"},
    }

    device = torch.device("cpu")

    # SingleValueBinarySelectionModel can be created with minimal params
    model = create_selection_model_from_dict(config, device)
    assert model.__class__.__name__ == "SingleValueBinarySelectionModel"


def test_create_selection_model_missing_hparams():
    """Test model creation with missing hparams key."""
    config = {
        "model_class": "SingleValueBinarySelectionModel"
        # Missing 'hparams' key - will use empty dict as default
    }

    device = torch.device("cpu")

    # Should use empty dict as default and create model with defaults
    # Note: This will still show model_type warning since hparams is empty
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = create_selection_model_from_dict(config, device)
    assert model.__class__.__name__ == "SingleValueBinarySelectionModel"


def test_create_selection_model_from_yaml():
    """Test model creation from YAML file."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    # Write to temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        yaml_path = f.name

    try:
        device = torch.device("cpu")
        model = create_selection_model_from_yaml(yaml_path, device)

        assert model.__class__.__name__ == "SingleValueBinarySelectionModel"
        assert model.model_type == "dasm"
        assert model.known_token_count == 21
        assert model.output_dim == 20
    finally:
        os.unlink(yaml_path)


def test_create_selection_model_from_yaml_transformer():
    """Test transformer model creation from YAML file."""
    config = {
        "model_class": "TransformerBinarySelectionModelWiggleAct",
        "hparams": {
            "model_type": "dasm",
            "known_token_count": 21,
            "output_dim": 20,
            "nhead": 2,
            "d_model_per_head": 4,
            "dim_feedforward": 512,
            "layer_count": 2,
            "dropout_prob": 0.0,
        },
    }

    # Write to temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        yaml_path = f.name

    try:
        device = torch.device("cpu")
        model = create_selection_model_from_yaml(yaml_path, device)

        assert model.__class__.__name__ == "TransformerBinarySelectionModelWiggleAct"
        assert model.nhead == 2
        assert model.d_model_per_head == 4
        assert model.pos_encoder.dropout.p == 0.0
    finally:
        os.unlink(yaml_path)


def test_create_selection_model_from_yaml_nonexistent_file():
    """Test error handling for nonexistent YAML file."""
    device = torch.device("cpu")

    with pytest.raises(FileNotFoundError):
        create_selection_model_from_yaml("nonexistent_file.yaml", device)


def test_model_device_placement():
    """Test that models are correctly moved to specified device."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    # Check that model parameters are on the correct device
    for param in model.parameters():
        assert param.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_device_placement_cuda():
    """Test model placement on CUDA device if available."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cuda")
    model = create_selection_model_from_dict(config, device)

    # Check that model parameters are on CUDA
    for param in model.parameters():
        assert param.device.type == "cuda"


def test_different_model_types():
    """Test creation of models with different model_type values."""
    model_types = ["dasm", "dnsm", "ddsm"]

    for model_type in model_types:
        config = {
            "model_class": "SingleValueBinarySelectionModel",
            "hparams": {
                "model_type": model_type,
                "known_token_count": 21,
                "output_dim": 20,
            },
        }

        device = torch.device("cpu")
        model = create_selection_model_from_dict(config, device)

        assert model.model_type == model_type


def test_various_hyperparameters():
    """Test model creation with various hyperparameter combinations."""
    config = {
        "model_class": "TransformerBinarySelectionModelWiggleAct",
        "hparams": {
            "model_type": "dasm",
            "known_token_count": 42,
            "output_dim": 30,
            "nhead": 6,
            "d_model_per_head": 8,
            "dim_feedforward": 2048,
            "layer_count": 4,
            "dropout_prob": 0.2,
        },
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    assert model.known_token_count == 42
    assert model.output_dim == 30
    assert model.nhead == 6
    assert model.d_model_per_head == 8
    assert model.dim_feedforward == 2048
    assert model.encoder.num_layers == 4
    assert model.pos_encoder.dropout.p == 0.2


# Tests for new functionality


def test_validate_config_schema():
    """Test configuration schema validation."""
    # Valid config
    config = {"model_class": "SingleValueBinarySelectionModel", "hparams": {}}
    validate_config_schema(config)  # Should not raise

    # Missing model_class
    with pytest.raises(ValueError, match="Missing required field: model_class"):
        validate_config_schema({})

    # Invalid model_class type
    with pytest.raises(ValueError, match="model_class must be a string"):
        validate_config_schema({"model_class": 123})

    # Invalid hparams type
    with pytest.raises(ValueError, match="hparams must be a dictionary"):
        validate_config_schema({"model_class": "Test", "hparams": "invalid"})


def test_list_available_models():
    """Test listing available model classes."""
    models = list_available_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert "SingleValueBinarySelectionModel" in models
    assert "TransformerBinarySelectionModelWiggleAct" in models
    assert all("SelectionModel" in name for name in models)


def test_list_available_presets():
    """Test listing available preset configurations."""
    presets = list_available_presets()

    assert isinstance(presets, list)
    assert len(presets) > 0
    assert "single_default" in presets
    assert "transformer_small" in presets
    assert "transformer_large" in presets
    assert "bidirectional_default" in presets
    assert "parent_independent_default" in presets


def test_get_model_info():
    """Test getting model information."""
    info = get_model_info("SingleValueBinarySelectionModel")

    assert info["class_name"] == "SingleValueBinarySelectionModel"
    assert "parameters" in info
    assert isinstance(info["parameters"], dict)

    # Test invalid model
    with pytest.raises(ValueError, match="Unknown model class"):
        get_model_info("NonexistentModel")


def test_describe_model():
    """Test model description functionality."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cpu")
    model = create_selection_model_from_dict(config, device)

    description = describe_model(model)

    assert description["model_class"] == "SingleValueBinarySelectionModel"
    assert isinstance(description["total_params"], int)
    assert isinstance(description["trainable_params"], int)
    assert description["device"] == "cpu"
    assert description["model_type"] == "dasm"


def test_create_model_from_preset():
    """Test creating models from presets."""
    device = torch.device("cpu")

    # Test available presets
    for preset_name in list_available_presets():
        model = create_model_from_preset(preset_name, device)
        assert model is not None
        assert next(model.parameters()).device == device

    # Test with overrides
    model = create_model_from_preset(
        "transformer_small", device, layer_count=3, dropout_prob=0.2
    )
    assert model.encoder.num_layers == 3
    assert model.pos_encoder.dropout.p == 0.2

    # Test invalid preset
    with pytest.raises(ValueError, match="Unknown preset"):
        create_model_from_preset("nonexistent", device)


def test_create_selection_model_from_file():
    """Test YAML file format detection."""
    config = {
        "model_class": "SingleValueBinarySelectionModel",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cpu")

    # Test YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        yaml_path = f.name

    try:
        model = create_selection_model_from_file(yaml_path, device)
        assert model.__class__.__name__ == "SingleValueBinarySelectionModel"
    finally:
        os.unlink(yaml_path)

    # Test unsupported format
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("test")
        txt_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file format"):
            create_selection_model_from_file(txt_path, device)
    finally:
        os.unlink(txt_path)


def test_transformer_validation():
    """Test transformer model parameter validation."""
    # Missing required transformer parameters
    config = {
        "model_class": "TransformerBinarySelectionModelWiggleAct",
        "hparams": {"model_type": "dasm", "known_token_count": 21, "output_dim": 20},
    }

    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Missing required transformer parameters"):
        create_selection_model_from_dict(config, device)


def test_improved_error_messages():
    """Test improved error messages with suggestions."""
    device = torch.device("cpu")

    # Test model initialization with invalid parameter
    config = {
        "model_class": "TransformerBinarySelectionModelWiggleAct",
        "hparams": {
            "model_type": "dasm",
            "known_token_count": 21,
            "output_dim": 20,
            "nhead": 4,
            "d_model_per_head": 4,
            "dim_feedforward": 512,
            "layer_count": 2,
            "invalid_param": "test",
        },
    }

    with pytest.raises(ValueError) as excinfo:
        create_selection_model_from_dict(config, device)

    error_message = str(excinfo.value)
    assert "Invalid hyperparameters" in error_message
    assert "get_model_info" in error_message


def test_preset_configs_validity():
    """Test that all preset configurations are valid."""
    device = torch.device("cpu")

    for preset_name in list_available_presets():
        # Load preset config
        preset_config = _load_preset_config(preset_name)

        # Each preset should create a valid model
        model = create_selection_model_from_dict(preset_config, device)
        assert model is not None

        # Description should work
        description = describe_model(model)
        assert description["model_class"] == preset_config["model_class"]
