from copy import deepcopy
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from netam import framework, models
from netam.common import pick_device, parameter_count_of_model


class Experiment:
    def __init__(self, site_count=500, epochs=1000, burrito_params=None):
        self.device = pick_device()

        if burrito_params is None:
            self.burrito_params = {
                "batch_size": 1024,
                "learning_rate": 0.1,
                "min_learning_rate": 1e-4,
                "l2_regularization_coeff": 1e-6,
            }
        else:
            self.burrito_params = burrito_params

        self.site_count = site_count
        self.epochs = epochs

    def build_model_instances(self, prename):
        model_instances_3 = {
            f"{prename}_cnn_sml": models.CNNModel(
                kmer_length=3,
                embedding_dim=6,
                filter_count=14,
                kernel_size=7,
                dropout_prob=0.1,
            ),
            f"{prename}_cnn_med_orig": models.CNNModel(
                kmer_length=3,
                embedding_dim=9,
                filter_count=9,
                kernel_size=11,
                dropout_prob=0.1,
            ),
            f"{prename}_cnn_med": models.CNNModel(
                kmer_length=3,
                embedding_dim=7,
                filter_count=16,
                kernel_size=9,
                dropout_prob=0.2,
            ),
            f"{prename}_cnn_lrg": models.CNNModel(
                kmer_length=3,
                embedding_dim=7,
                filter_count=19,
                kernel_size=11,
                dropout_prob=0.3,
            ),
            f"{prename}_cnn_4k": models.CNNModel(
                kmer_length=3,
                embedding_dim=12,
                filter_count=14,
                kernel_size=17,
                dropout_prob=0.1,
            ),
            f"{prename}_cnn_4k_k13": models.CNNModel(
                kmer_length=3,
                embedding_dim=12,
                filter_count=20,
                kernel_size=13,
                dropout_prob=0.3,
            ),
            f"{prename}_cnn_8k": models.CNNModel(
                kmer_length=3,
                embedding_dim=14,
                filter_count=25,
                kernel_size=15,
                dropout_prob=0.0,
            ),
        }

        model_instances_5 = {
            f"{prename}_fivemer": models.FivemerModel(kmer_length=5),
            f"{prename}_shmoof": models.SHMoofModel(
                kmer_length=5, site_count=self.site_count
            ),
        }
        return model_instances_3, model_instances_5

    def data_by_kmer_length_of(self, data_df):
        data_dict = {
            kmer_length: framework.SHMoofDataset(
                data_df, kmer_length=kmer_length, site_count=self.site_count
            )
            for kmer_length in [1, 3, 5]
        }
        for data in data_dict.values():
            data.to(self.device)
        return data_dict

    def train_or_load(
        self,
        model_name,
        model,
        train_dataset,
        val_dataset,
        training_params,
        pretrained_dir="../pretrained",
    ):
        crepe_prefix = f"{pretrained_dir}/{model_name}"

        if framework.crepe_exists(crepe_prefix):
            print(f"\tLoading pre-trained {model_name}...")
            crepe = framework.load_crepe(crepe_prefix)
            assert crepe.model.hyperparameters == model.hyperparameters
            for key in training_params:
                assert training_params[key] == crepe.training_hyperparameters[key]
            crepe.model.to(self.device)
            return crepe.model

        # else:
        print(f"\tTraining {model_name}...")
        train_dataset.to(self.device)
        val_dataset.to(self.device)
        model.to(self.device)

        our_burrito_params = deepcopy(self.burrito_params)
        our_burrito_params.update(training_params)
        burrito = framework.Burrito(
            train_dataset, val_dataset, model, verbose=False, **our_burrito_params
        )
        train_history = burrito.multi_train(epochs=self.epochs)
        Path(pretrained_dir).mkdir(parents=True, exist_ok=True)
        burrito.save_crepe(crepe_prefix)

        return model

    @staticmethod
    def experiment_df_of(model_instances, train_dataset, val_dataset):
        row_list = []

        for model_name, model in model_instances.items():
            row_list.append(
                {
                    "model_name": model_name,
                    "model": model,
                    "parameter_count": parameter_count_of_model(model),
                    "kmer_length": model.kmer_length,
                    "train_dataset": train_dataset,
                    "val_dataset": val_dataset,
                }
            )

        return pd.DataFrame(row_list)

    def build_experiment_df(
        self,
        prename,
        train_data_by_kmer_length,
        val_data_by_kmer_length,
        training_params_by_model_name,
    ):
        model_instances_3, model_instances_5 = self.build_model_instances(prename)
        experiment_df = pd.concat(
            [
                self.experiment_df_of(
                    model_instances_3,
                    train_data_by_kmer_length[3],
                    val_data_by_kmer_length[3],
                ),
                self.experiment_df_of(
                    model_instances_5,
                    train_data_by_kmer_length[5],
                    val_data_by_kmer_length[5],
                ),
            ],
            ignore_index=True,
        )
        experiment_df["training_params"] = [
            training_params_by_model_name.get(row["model_name"], {})
            for _, row in experiment_df.iterrows()
        ]
        return experiment_df

    def train_experiment_df(self, experiment_df, pretrained_dir="../pretrained"):
        for index, row in experiment_df.iterrows():
            model_name = row["model_name"]
            model_instance = row["model"]
            train_dataset = row["train_dataset"]
            val_dataset = row["val_dataset"]
            training_params = row["training_params"]

            trained_model = self.train_or_load(
                model_name,
                model_instance,
                train_dataset,
                val_dataset,
                training_params,
                pretrained_dir=pretrained_dir,
            )
            experiment_df.at[index, "model"] = trained_model

    def calculate_loss(self, model, dataset):
        model.eval()
        burrito = framework.Burrito(
            dataset, dataset, model, verbose=False, **self.burrito_params
        )
        loss = burrito.evaluate()
        return loss

    def loss_of_dataset_dict(self, experiment_df, dataset_dict):
        return [
            self.calculate_loss(row["model"], dataset_dict[row["kmer_length"]])
            for _, row in experiment_df.iterrows()
        ]


def plot_loss_difference(expt_df, baseline_model_name):
    df = expt_df
    assert baseline_model_name in df["model_name"].values, "Baseline model not found"
    # Identify loss columns (ending with '_loss')
    loss_columns = [col for col in df.columns if col.endswith("_loss")]
    assert len(loss_columns) > 0, "No loss columns found"

    # Calculate differences from the baseline model for each loss type
    for loss_col in loss_columns:
        baseline_loss = df[df["model_name"] == baseline_model_name][loss_col].values[0]
        df[f"{loss_col}_diff"] = df[loss_col] - baseline_loss

    # Filter out the baseline model and sort by parameter count
    df = df[df["model_name"] != baseline_model_name]
    df = df.sort_values(by="parameter_count")

    # Prepare data for plotting
    melted_df = pd.melt(
        df,
        id_vars=["model_name", "parameter_count"],
        value_vars=[f"{col}_diff" for col in loss_columns],
        var_name="Loss Type",
        value_name="Loss Difference",
    )

    # Create a separate plot for each loss type
    n_loss_types = len(loss_columns)
    fig, axes = plt.subplots(
        n_loss_types, 1, figsize=(8, 4 * n_loss_types), squeeze=False
    )

    for i, loss_type in enumerate(loss_columns):
        sns.barplot(
            data=melted_df[melted_df["Loss Type"] == f"{loss_type}_diff"],
            x="Loss Difference",
            y="model_name",
            ax=axes[i, 0],
        )
        axes[i, 0].set_title(loss_type.replace("_loss", "").replace("_", " ").title())
        axes[i, 0].axvline(0, color="black", linewidth=1)  # Add vertical line at zero

    plt.tight_layout()
    plt.show()
