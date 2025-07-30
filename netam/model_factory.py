"""Model factory functions for creating netam models.

This module provides factory functions to create netam selection models from
configuration dictionaries and YAML files. Supports all model types including:
- SingleValueBinarySelectionModel
- ParentIndependentBinarySelectionModel
- TransformerBinarySelectionModelLinAct
- TransformerBinarySelectionModelWiggleAct
- BidirectionalTransformerBinarySelectionModel

Example usage:
    >>> import netam
    >>> config = {'model_class': 'SingleValueBinarySelectionModel',
    ...           'hparams': {'model_type': 'dasm'}}
    >>> model = netam.create_selection_model_from_dict(config, torch.device('cpu'))
"""

import torch
import inspect
import yaml
from typing import Dict, Any, List
from pathlib import Path
from importlib.resources import files

MODEL_CONFIGS_DIR = files("netam") / "model_configs"


def _load_preset_config(preset_name: str) -> Dict[str, Any]:
    """Load a preset configuration from the model_configs directory.

    Args:
        preset_name: Name of the preset (filename without .yaml extension)

    Returns:
        Configuration dictionary loaded from YAML file

    Raises:
        FileNotFoundError: If preset file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    """
    preset_path = MODEL_CONFIGS_DIR / f"{preset_name}.yaml"
    if not preset_path.exists():
        available = list_available_presets()
        raise ValueError(
            f"Unknown preset: {preset_name}. Available presets: {available}"
        )

    with open(preset_path, "r") as f:
        return yaml.safe_load(f)


def list_available_presets() -> List[str]:
    """List all available preset configurations.

    Returns:
        List of preset names (filenames without .yaml extension)
    """
    if not MODEL_CONFIGS_DIR.exists():
        return []

    return sorted([p.stem for p in MODEL_CONFIGS_DIR.glob("*.yaml") if p.is_file()])


def validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate configuration has required fields and types.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = ["model_class"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(config["model_class"], str):
        raise ValueError("model_class must be a string")

    if "hparams" in config and not isinstance(config["hparams"], dict):
        raise ValueError("hparams must be a dictionary")


def _validate_hparams(model_class_name: str, hparams: Dict[str, Any]) -> None:
    """Validate hyperparameters for specific model types.

    Args:
        model_class_name: Name of the model class
        hparams: Hyperparameters dictionary

    Raises:
        ValueError: If required parameters are missing for model type
    """
    transformer_models = {
        "TransformerBinarySelectionModelLinAct",
        "TransformerBinarySelectionModelWiggleAct",
        "BidirectionalTransformerBinarySelectionModel",
    }

    if model_class_name in transformer_models:
        required = ["nhead", "d_model_per_head", "dim_feedforward", "layer_count"]
        missing = [param for param in required if param not in hparams]
        if missing:
            raise ValueError(
                f"Missing required transformer parameters for {model_class_name}: {missing}. "
                f"Consider using a preset like 'transformer_small' or 'transformer_large'."
            )


def list_available_models() -> List[str]:
    """Return list of all available selection model classes.

    Returns:
        List of model class names that can be used in configurations
    """
    import netam.models as models

    return sorted(
        [
            name
            for name in dir(models)
            if ("SelectionModel" in name or "BinarySelectionModel" in name)
            and not name.startswith("_")
            and name != "AbstractBinarySelectionModel"
        ]
    )


def get_model_info(model_class_name: str) -> Dict[str, Any]:
    """Get information about a model class including required parameters.

    Args:
        model_class_name: Name of the model class

    Returns:
        Dictionary with model information including signature and docstring

    Raises:
        ValueError: If model class is not found
    """
    import netam.models as models

    if not hasattr(models, model_class_name):
        available = list_available_models()
        raise ValueError(
            f"Unknown model class: {model_class_name}. "
            f"Available model classes: {available}"
        )

    model_class = getattr(models, model_class_name)
    sig = inspect.signature(model_class.__init__)

    return {
        "class_name": model_class_name,
        "docstring": model_class.__doc__,
        "parameters": {
            name: {
                "annotation": (
                    str(param.annotation) if param.annotation != param.empty else None
                ),
                "default": param.default if param.default != param.empty else None,
            }
            for name, param in sig.parameters.items()
            if name != "self"
        },
    }


def describe_model(model) -> Dict[str, Any]:
    """Return model architecture summary and parameter counts.

    Args:
        model: PyTorch model instance

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_class": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": (
            str(next(model.parameters()).device)
            if total_params > 0
            else "no_parameters"
        ),
        "model_type": getattr(model, "model_type", None),
    }


def create_model_from_preset(
    preset_name: str, device: torch.device, **overrides
) -> torch.nn.Module:
    """Create model from predefined preset with optional parameter overrides.

    Args:
        preset_name: Name of the preset configuration
        device: Target device for the model
        **overrides: Additional hyperparameters to override preset values

    Returns:
        Initialized selection model moved to device

    Raises:
        ValueError: If preset name is not found

    Example:
        >>> model = create_model_from_preset('transformer_small', torch.device('cpu'),
        ...                                   layer_count=4, dropout_prob=0.2)
    """
    # Load preset configuration from file
    config = _load_preset_config(preset_name)

    # Apply overrides to hparams
    if overrides:
        config["hparams"] = config.get("hparams", {}).copy()
        config["hparams"].update(overrides)

    return create_selection_model_from_dict(config, device)


def create_selection_model_from_dict(
    config: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """Create a selection model from configuration dictionary.

    Args:
        config: Configuration dictionary containing:
            - model_class (str): Model class name from netam.models
            - hparams (dict, optional): Model hyperparameters including:
                * model_type (str): One of 'dasm', 'dnsm', 'ddsm'
                * known_token_count (int): Number of known tokens (default: 21)
                * output_dim (int): Output dimension (default: 20)
                * For transformer models: nhead, d_model_per_head, etc.
        device: PyTorch device for model placement

    Returns:
        torch.nn.Module: Initialized selection model on specified device

    Raises:
        ValueError: If model_class is unknown or invalid, or required parameters missing
        TypeError: If model initialization fails due to parameter issues

    Example:
        >>> config = {
        ...     'model_class': 'SingleValueBinarySelectionModel',
        ...     'hparams': {
        ...         'model_type': 'dasm',
        ...         'known_token_count': 21,
        ...         'output_dim': 20
        ...     }
        ... }
        >>> device = torch.device('cpu')
        >>> model = create_selection_model_from_dict(config, device)
    """
    import netam.models as models

    # Validate configuration schema
    validate_config_schema(config)

    model_class_name = config["model_class"]
    hparams = config.get("hparams", {})

    # Validate hyperparameters for model type
    _validate_hparams(model_class_name, hparams)

    # Get model class using module attributes
    if not hasattr(models, model_class_name):
        available = list_available_models()
        raise ValueError(
            f"Unknown model class: {model_class_name}. "
            f"Available model classes: {available}"
        )

    model_class = getattr(models, model_class_name)

    try:
        model = model_class(**hparams)
    except TypeError as e:
        raise ValueError(
            f"Invalid hyperparameters for {model_class_name}: {e}. "
            f"Check the model's __init__ signature for required parameters. "
            f"Use get_model_info('{model_class_name}') for parameter details."
        ) from e

    return model.to(device)


def create_selection_model_from_yaml(
    yaml_path: str, device: torch.device
) -> torch.nn.Module:
    """Create a selection model from YAML configuration file.

    Args:
        yaml_path: Path to YAML file containing model configuration
        device: Target device for the model

    Returns:
        torch.nn.Module: Initialized selection model moved to device

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML file is malformed

    Example:
        >>> device = torch.device('cpu')
        >>> model = create_selection_model_from_yaml('model_config.yaml', device)
    """
    import yaml

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    return create_selection_model_from_dict(config, device)


def create_selection_model_from_file(
    config_path: str, device: torch.device
) -> torch.nn.Module:
    """Create model from YAML configuration file.

    Args:
        config_path: Path to YAML configuration file (.yaml or .yml)
        device: Target device for the model

    Returns:
        torch.nn.Module: Initialized selection model moved to device

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If configuration file doesn't exist

    Example:
        >>> device = torch.device('cpu')
        >>> model = create_selection_model_from_file('config.yaml', device)
    """
    path = Path(config_path)

    if path.suffix.lower() in [".yml", ".yaml"]:
        return create_selection_model_from_yaml(config_path, device)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported formats: .yaml, .yml"
        )
