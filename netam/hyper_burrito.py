"""HyperBurrito class for optimizing hyperparameters."""

from abc import ABC, abstractmethod
from datetime import datetime
import inspect

import optuna

from netam.common import parameter_count_of_model
from netam.framework import SHMBurrito


def filter_kwargs(func, kwargs):
    """Filter kwargs to only those that the function accepts."""
    # Get the parameters of the function
    sig = inspect.signature(func)
    func_params = sig.parameters

    # Filter kwargs to only those that the function accepts
    return {k: v for k, v in kwargs.items() if k in func_params}


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class HyperBurrito(ABC):
    """A class to optimize hyperparameters."""

    def __init__(
        self,
        device,
        train_dataset,
        val_dataset,
        model_class,
        epochs=100,
    ):
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        train_dataset.to(self.device)
        val_dataset.to(self.device)
        self.model_class = model_class
        self.epochs = epochs

    @abstractmethod
    def burrito_of_model(self, model, **kwargs):
        pass

    def optuna_objective(
        self,
        trial,
        int_params,
        cat_params,
        float_params,
        log_float_params,
        fixed_hyperparams=None,
    ):
        """Optuna objective function.

        Return validation loss unless the model has more parameters than allowed
        in fixed_hyperparams["max_parameter_count"], in which case return 1e9.

        Note that if a parameter appears in one of the xxx_params dictionaries
        used for sampling as well as the fixed_hyperparams dictionary, the
        sampled value will be used.

        Args:
            trial (optuna.Trial): Optuna trial object.
            int_params (dict): Dictionary of integer parameters to optimize.
            cat_params (dict): Dictionary of categorical parameters to optimize.
            float_params (dict): Dictionary of float parameters to optimize on a linear scale.
            log_float_params (dict): Dictionary of float parameters to optimize on a log scale.
            fixed_hyperparams (dict, optional): Dictionary of fixed hyperparameters. Defaults to None.

        Returns:
            float: Validation loss or 1e9 if the model has too many parameters.
        """
        if fixed_hyperparams is not None:
            for key, value in fixed_hyperparams.items():
                trial.set_user_attr(key, value)

        hyperparams = fixed_hyperparams or {}

        for param_name, choices in cat_params.items():
            hyperparams[param_name] = trial.suggest_categorical(param_name, choices)

        for param_name, (low, high) in int_params.items():
            hyperparams[param_name] = trial.suggest_int(param_name, low, high)

        for param_name, (low, high) in float_params.items():
            hyperparams[param_name] = trial.suggest_float(param_name, low, high)

        for param_name, (low, high) in log_float_params.items():
            hyperparams[param_name] = trial.suggest_float(
                param_name, low, high, log=True
            )

        model_hyperparams = filter_kwargs(self.model_class, hyperparams)
        model = self.model_class(**model_hyperparams)
        model.to(self.device)

        if hyperparams is not None and "max_parameter_count" in hyperparams:
            parameter_count = parameter_count_of_model(model)
            # if parameter_count is not in the range between hyperparams["min_parameter_count"] and hyperparams["max_parameter_count"]:
            if parameter_count not in range(
                hyperparams["min_parameter_count"],
                hyperparams["max_parameter_count"] + 1,
            ):
                range_str = f"[{hyperparams['min_parameter_count']}, {hyperparams['max_parameter_count']}]"
                print(
                    f"Trial rejected. Model has {parameter_count} parameters, not in {range_str}]."
                )
                return 1e9

        burrito_hyperparams = filter_kwargs(self.burrito_of_model, hyperparams)
        print("burrito_hypers:", burrito_hyperparams)
        burrito = self.burrito_of_model(model, **burrito_hyperparams)

        losses = burrito.joint_train(epochs=self.epochs)

        return losses["val_loss"].min()

    def optuna_optimize(
        self,
        n_trials,
        cat_params,
        int_params,
        float_params,
        log_float_params,
        fixed_hyperparams=None,
        study_name=None,
    ):
        storage_url = "sqlite:///_ignore/optuna.db"
        if study_name is None:
            study_name = f"study_{self.model_class.__name__}"

        # Create or load the study
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: self.optuna_objective(
                trial,
                int_params,
                cat_params,
                float_params,
                log_float_params,
                fixed_hyperparams,
            ),
            n_trials=n_trials,
        )
        best_hyperparams = study.best_params
        best_score = study.best_value
        print(f"Best Hyperparameters: {best_hyperparams}")
        print(f"Best Validation Loss: {best_score}")

        output_path = (
            f"_ignore/optuna_{self.model_class.__name__}_{timestamp_str()}.csv"
        )
        trial_data = study.trials_dataframe()
        trial_data.to_csv(output_path, index=False)


class SHMHyperBurrito(HyperBurrito):
    # Note that we have to write the args out explicitly because we use some
    # magic to filter kwargs in the optuna_objective method.
    def burrito_of_model(
        self,
        model,
        batch_size=1024,
        learning_rate=0.1,
        min_learning_rate=1e-4,
        weight_decay=1e-6,
    ):
        burrito = SHMBurrito(
            self.train_dataset,
            self.val_dataset,
            model,
            batch_size=batch_size,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            weight_decay=weight_decay,
        )
        return burrito
