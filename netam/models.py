from abc import ABC, abstractmethod
import math
import warnings
from datetime import datetime, timezone

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from netam import molevol
from netam.hit_class import apply_multihit_correction
from netam import sequences
from netam.sequences import MAX_AA_TOKEN_IDX
from netam.common import (
    chunk_function,
    zap_predictions_along_diagonal,
    clamp_probability,
    clamp_probability_above_only,
)
from netam.sequences import (
    generate_kmers,
    aa_mask_tensor_of,
    encode_sequences,
    aa_idx_tensor_of_str_ambig,
    PositionalEncoding,
    split_heavy_light_model_outputs,
    AA_PADDING_TOKEN,
    flatten_codon_idxs,
)
from typing import Tuple

# If this changes, we need to update old models that may not have neutral model
# in their metadata
DEFAULT_NEUTRAL_MODEL = "ThriftyHumV0.2-59"
DEFAULT_MULTIHIT_MODEL = "ThriftyHumV0.2-59-hc-tangshm"


warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.modules.transformer"
)


class ModelBase(nn.Module):
    def reinitialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.TransformerEncoder):
                for sublayer in layer.modules():
                    if isinstance(sublayer, nn.Linear):
                        nn.init.kaiming_normal_(sublayer.weight, nonlinearity="relu")
                        if sublayer.bias is not None:
                            nn.init.constant_(sublayer.bias, 0)
            elif isinstance(layer, nn.Dropout):
                pass
            elif hasattr(layer, "reinitialize_weights"):
                layer.reinitialize_weights()
            else:
                raise ValueError(f"Unrecognized layer type: {type(layer)}")

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        """Freeze all parameters in the model, disabling gradient computations."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the model, enabling gradient computations."""
        for param in self.parameters():
            param.requires_grad = True

    def predictions_of_sequences(self, sequences, encoder=None, **kwargs):
        return self.evaluate_sequences(sequences, encoder=encoder, **kwargs)

    @chunk_function(first_chunkable_idx=1, progress_bar_name="Evaluating model")
    def evaluate_sequences(self, sequences, encoder=None, chunk_size=2048):
        if encoder is None:
            raise ValueError("An encoder must be provided.")
        encoded_parents, masks, wt_base_modifiers = encode_sequences(sequences, encoder)
        encoded_parents = encoded_parents.to(self.device)
        masks = masks.to(self.device)
        wt_base_modifiers = wt_base_modifiers.to(self.device)
        with torch.no_grad():
            outputs = self(encoded_parents, masks, wt_base_modifiers)
            return tuple(t.detach().cpu() for t in outputs)

    def represent_aa_str(self, *args, **kwargs):
        raise NotImplementedError("represent_aa_str is implemented on D*SM models only")


class KmerModel(ModelBase):
    def __init__(self, kmer_length):
        super().__init__()
        self.kmer_length = kmer_length
        self.all_kmers = generate_kmers(kmer_length)
        self.kmer_count = len(self.all_kmers)

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
        }


class FivemerModel(KmerModel):
    def __init__(self):
        super().__init__(kmer_length=5)
        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze(-1)
        rates = torch.exp(log_kmer_rates * masks)
        return rates

    def adjust_rate_bias_by(self, log_adjustment_factor):
        with torch.no_grad():
            self.kmer_embedding.weight.data += log_adjustment_factor

    @property
    def kmer_rates(self):
        return torch.exp(self.kmer_embedding.weight).squeeze()


class RSFivemerModel(KmerModel):
    def __init__(self, kmer_length=5):
        assert kmer_length == 5
        super().__init__(kmer_length=5)
        self.r_kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.s_kmer_embedding = nn.Embedding(self.kmer_count, 4)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        log_kmer_rates = self.r_kmer_embedding(encoded_parents).squeeze(-1)
        rates = torch.exp(log_kmer_rates * masks)
        csp_logits = self.s_kmer_embedding(encoded_parents)
        # When we have an N, set all the CSP logits to 0, resulting in a uniform
        # prediction. There is nothing to predict here.
        csp_logits *= masks.unsqueeze(-1)
        # As described elsewhere, this makes the WT base have a probability of 0
        # after softmax.
        csp_logits += wt_base_modifier
        return rates, csp_logits

    def adjust_rate_bias_by(self, log_adjustment_factor):
        with torch.no_grad():
            self.r_kmer_embedding.weight.data += log_adjustment_factor

    @property
    def kmer_rates(self):
        return torch.exp(self.r_kmer_embedding.weight).squeeze()


class SHMoofModel(KmerModel):
    def __init__(self, kmer_length, site_count):
        super().__init__(kmer_length)
        self.site_count = site_count
        self.kmer_embedding = nn.Embedding(self.kmer_count, 1)
        self.log_site_rates = nn.Embedding(self.site_count, 1)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        log_kmer_rates = self.kmer_embedding(encoded_parents).squeeze(-1)
        # When we transpose we get a tensor of shape [site_count, 1], which will broadcast
        # to the shape of log_kmer_rates, repeating over the batch dimension.
        log_site_rates = self.log_site_rates.weight.T
        # Rates are the product of kmer and site rates.
        rates = torch.exp((log_kmer_rates + log_site_rates) * masks)
        return rates

    def adjust_rate_bias_by(self, log_adjustment_factor):
        with torch.no_grad():
            self.kmer_embedding.weight.data += log_adjustment_factor / 2.0
            self.log_site_rates.weight.data += log_adjustment_factor / 2.0

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
            "site_count": self.site_count,
        }

    @property
    def kmer_rates(self):
        return torch.exp(self.kmer_embedding.weight).squeeze()

    @property
    def site_rates(self):
        return torch.exp(self.log_site_rates.weight).squeeze()

    def write_shmoof_output(self, out_dir):
        # Extract k-mer (motif) mutabilities
        kmer_rates = self.kmer_rates.detach().numpy().flatten()
        motif_mutabilities = pd.DataFrame(
            {
                "Motif": self.all_kmers,
                "Mutability": kmer_rates,
            }
        )
        motif_mutabilities.to_csv(
            f"{out_dir}/motif_mutabilities.tsv", sep="\t", index=False
        )

        # Extract site mutabilities
        site_mutabilities = self.site_rates.detach().numpy().flatten()
        site_mutabilities_df = pd.DataFrame(
            {
                "Position": range(1, len(site_mutabilities) + 1),
                "Mutability": site_mutabilities,
            }
        )
        site_mutabilities_df.to_csv(
            f"{out_dir}/site_mutabilities.tsv", sep="\t", index=False
        )


class RSSHMoofModel(KmerModel):
    def __init__(self, kmer_length, site_count):
        super().__init__(kmer_length)
        self.site_count = site_count
        self.kmer_embedding = nn.Embedding(self.kmer_count, 4)
        self.log_site_rates = nn.Embedding(self.site_count, 1)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        log_kmer_rates_per_base = self.kmer_embedding(encoded_parents)
        # Set WT base to have rate of 0 in log space.
        log_kmer_rates_per_base += wt_base_modifier
        # Here we are summing over the per-base dimensions to get the per-kmer
        # rates. We want to do so in linear, not log space.
        log_kmer_rates = torch.logsumexp(log_kmer_rates_per_base, dim=-1)
        assert log_kmer_rates.shape == (
            encoded_parents.size(0),
            encoded_parents.size(1),
        )

        # When we transpose we get a tensor of shape [site_count, 1], which will broadcast
        # to the shape of log_kmer_rates, repeating over the batch dimension.
        log_site_rates = self.log_site_rates.weight.T
        # Rates are the product of kmer and site rates.
        rates = torch.exp((log_kmer_rates + log_site_rates) * masks)

        csp_logits = log_kmer_rates_per_base * masks.unsqueeze(-1)
        csp_logits += wt_base_modifier

        return rates, csp_logits

    def adjust_rate_bias_by(self, log_adjustment_factor):
        with torch.no_grad():
            self.kmer_embedding.weight.data += log_adjustment_factor / 2.0
            self.log_site_rates.weight.data += log_adjustment_factor / 2.0

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
            "site_count": self.site_count,
        }


class CNNModel(KmerModel):
    """This is a CNN model that uses k-mers as input and trains an embedding layer."""

    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1
    ):
        super().__init__(kmer_length)
        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_features=filter_count, out_features=1)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # Transpose back for Linear layer
        log_rates = self.linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)
        return rates

    def adjust_rate_bias_by(self, log_adjustment_factor):
        with torch.no_grad():
            self.linear.bias.data += log_adjustment_factor

    @property
    def hyperparameters(self):
        return {
            "kmer_length": self.kmer_length,
            "embedding_dim": self.kmer_embedding.embedding_dim,
            "filter_count": self.conv.out_channels,
            "kernel_size": self.conv.kernel_size[0],
            "dropout_prob": self.dropout.p,
        }


class CNNPEModel(CNNModel):
    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
    ):
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout_prob)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = self.pos_encoder(kmer_embeds)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)  # Transpose back for Linear layer
        log_rates = self.linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)
        return rates


class CNN1merModel(CNNModel):
    """This is a CNN model that uses individual bases as input and does not train an
    embedding layer."""

    def __init__(self, filter_count, kernel_size, dropout_prob=0.1):
        # Fixed embedding_dim because there are only 4 bases.
        embedding_dim = 5
        kmer_length = 1
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )
        # Here's how we adapt the model to use individual bases as input rather
        # than trainable kmer embeddings.
        identity_matrix = torch.eye(embedding_dim)
        self.kmer_embedding.weight = nn.Parameter(identity_matrix, requires_grad=False)


class RSCNNModel(CNNModel, ABC):
    """The base class for all RSCNN models. These are CNN models that predict both rates
    and CSP logits.

    They differ in how much they share the weights between the r_ components that
    predict rates, and the s_ components that predict CSP logits.

    See
    https://github.com/matsengrp/netam/pull/9#issuecomment-1939097576
    for diagrams about the various models.
    """

    @abstractmethod
    def forward(self, encoded_parents, masks, wt_base_modifier):
        pass

    def adjust_rate_bias_by(self, log_adjustment_factor):
        with torch.no_grad():
            self.r_linear.bias.data += log_adjustment_factor


class JoinedRSCNNModel(RSCNNModel):
    """This model shares everything except the final linear layers."""

    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1
    ):
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )
        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.r_linear = nn.Linear(in_features=filter_count, out_features=1)
        self.s_linear = nn.Linear(in_features=filter_count, out_features=4)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)  # Transpose for Conv1D
        conv_out = F.relu(self.conv(kmer_embeds))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.permute(0, 2, 1)

        log_rates = self.r_linear(conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)

        csp_logits = self.s_linear(conv_out)
        csp_logits *= masks.unsqueeze(-1)
        csp_logits += wt_base_modifier

        return rates, csp_logits


class HybridRSCNNModel(RSCNNModel):
    """This model shares the kmer_embedding only."""

    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1
    ):
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )

        self.kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        # Duplicate the layers for the r_ component
        self.r_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.r_dropout = nn.Dropout(dropout_prob)
        self.r_linear = nn.Linear(in_features=filter_count, out_features=1)

        self.s_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.s_dropout = nn.Dropout(dropout_prob)
        self.s_linear = nn.Linear(in_features=filter_count, out_features=4)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        kmer_embeds = self.kmer_embedding(encoded_parents)
        kmer_embeds = kmer_embeds.permute(0, 2, 1)
        r_conv_out = F.relu(self.r_conv(kmer_embeds))
        r_conv_out = self.r_dropout(r_conv_out)
        r_conv_out = r_conv_out.permute(0, 2, 1)
        s_conv_out = F.relu(self.s_conv(kmer_embeds))
        s_conv_out = self.s_dropout(s_conv_out)
        s_conv_out = s_conv_out.permute(0, 2, 1)

        log_rates = self.r_linear(r_conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)

        csp_logits = self.s_linear(s_conv_out)
        csp_logits *= masks.unsqueeze(-1)
        csp_logits += wt_base_modifier

        return rates, csp_logits


class IndepRSCNNModel(RSCNNModel):
    """This model does not share any weights between the r_ and s_ components."""

    def __init__(
        self, kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob=0.1
    ):
        super().__init__(
            kmer_length, embedding_dim, filter_count, kernel_size, dropout_prob
        )

        # Duplicate the layers for the r_ component
        self.r_kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.r_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.r_dropout = nn.Dropout(dropout_prob)
        self.r_linear = nn.Linear(in_features=filter_count, out_features=1)

        # Duplicate the layers for the s_ component
        self.s_kmer_embedding = nn.Embedding(self.kmer_count, embedding_dim)
        self.s_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filter_count,
            kernel_size=kernel_size,
            padding="same",
        )
        self.s_dropout = nn.Dropout(dropout_prob)
        self.s_linear = nn.Linear(in_features=filter_count, out_features=4)

    def forward(self, encoded_parents, masks, wt_base_modifier):
        # Process for r_ component
        r_kmer_embeds = self.r_kmer_embedding(encoded_parents)
        r_kmer_embeds = r_kmer_embeds.permute(0, 2, 1)
        r_conv_out = F.relu(self.r_conv(r_kmer_embeds))
        r_conv_out = self.r_dropout(r_conv_out)
        r_conv_out = r_conv_out.permute(0, 2, 1)

        log_rates = self.r_linear(r_conv_out).squeeze(-1)
        rates = torch.exp(log_rates * masks)

        # Process for s_ component
        s_kmer_embeds = self.s_kmer_embedding(encoded_parents)
        s_kmer_embeds = s_kmer_embeds.permute(0, 2, 1)
        s_conv_out = F.relu(self.s_conv(s_kmer_embeds))
        s_conv_out = self.s_dropout(s_conv_out)
        s_conv_out = s_conv_out.permute(0, 2, 1)

        csp_logits = self.s_linear(s_conv_out)
        csp_logits *= masks.unsqueeze(-1)
        csp_logits += wt_base_modifier

        return rates, csp_logits


# Issue #8
class WrapperHyperparameters:
    def __init__(self, base_model_hyperparameters, site_count):
        self.base_model_hyperparameters = base_model_hyperparameters
        self.site_count = site_count

    def __getitem__(self, key):
        if key in self.base_model_hyperparameters:
            return self.base_model_hyperparameters[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in hyperparameters.")

    def __str__(self):
        hyperparameters_dict = {key: getattr(self, key) for key in self.__dict__}
        return str(hyperparameters_dict)


class PersiteWrapper(ModelBase):
    """This wraps another model, but adds a per-site rate component."""

    def __init__(self, base_model, site_count):
        super().__init__()
        self.base_model = base_model
        self.site_count = site_count
        self.log_site_rates = nn.Embedding(self.site_count, 1)
        self._hyperparameters = WrapperHyperparameters(
            self.base_model.hyperparameters, self.site_count
        )

    def forward(self, encoded_parents, masks):
        base_model_rates = self.base_model(encoded_parents, masks)
        log_site_rates = self.log_site_rates.weight.T
        rates = base_model_rates * torch.exp(log_site_rates)
        return rates

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def site_rates(self):
        return torch.exp(self.log_site_rates.weight).squeeze()


class AbstractBinarySelectionModel(ABC, nn.Module):
    """A transformer-based model for binary selection.

    This is a model that takes in a batch of  index-encoded sequences and
    outputs a vector that represents the log level of selection for each amino
    acid site, which after exponentiating is a multiplier on the probability of
    an amino-acid substitution at that site.

    Various submodels are implemented as subclasses of this class:

    * LinAct: No activation function after the transformer.
    * WiggleAct: Activation that slopes to 0 at -inf and grows sub-linearly as x increases.

    See forward() for details.
    """

    def __init__(
        self,
        output_dim: int = 1,
        known_token_count: int = MAX_AA_TOKEN_IDX + 1,
        neutral_model_name: str = DEFAULT_NEUTRAL_MODEL,
        multihit_model_name: str = DEFAULT_MULTIHIT_MODEL,
        train_timestamp: str = None,
        model_type: str = None,
    ):
        super().__init__()
        if train_timestamp is None:
            train_timestamp = datetime.now(timezone.utc).isoformat(timespec="minutes")
        self.train_timestamp = train_timestamp
        self.output_dim = output_dim
        self.known_token_count = known_token_count
        self.neutral_model_name = neutral_model_name
        self.multihit_model_name = multihit_model_name
        if model_type is None:
            warnings.warn(
                "model_type should be specified. Either 'dasm', 'dnsm', or 'ddsm' expected."
            )
        self.model_type = model_type

    @property
    def hyperparameters(self):
        return {
            "output_dim": self.output_dim,
            "known_token_count": self.known_token_count,
            "neutral_model_name": self.neutral_model_name,
            "multihit_model_name": self.multihit_model_name,
            "train_timestamp": self.train_timestamp,
            "model_type": self.model_type,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def predictions_of_sequences(self, sequences, **kwargs):
        """Predict the selection factors for a list of amino acid sequences.

        For models which output a prediction for each possible target amino acid,
        wildtype amino acids are set to NaN, reflecting the fact that the model's
        predictions for wildtype amino acids are unconstrained in training and therefore
        meaningless.
        """
        return self.evaluate_sequences(sequences, **kwargs)

    def evaluate_sequences(self, sequences: list[str], **kwargs) -> Tensor:
        return list(self.selection_factors_of_aa_str(seq) for seq in sequences)

    def prepare_aa_str(self, heavy_chain, light_chain):
        """Prepare a pair of amino acid sequences for input to the model.

        Returns:
            A tuple of two tensors, the first being the index-encoded parent amino acid
            sequences and the second being the mask tensor.
            Although both represent a single sequence, they include a per-sequence first
            dimension for direct ingestion by the model.
        """
        aa_str, added_indices = sequences.prepare_heavy_light_pair(
            heavy_chain,
            light_chain,
            self.hyperparameters["known_token_count"],
            is_nt=False,
        )
        aa_idxs = aa_idx_tensor_of_str_ambig(aa_str)
        if torch.any(aa_idxs >= self.hyperparameters["known_token_count"]):
            raise ValueError(
                "Provided sequence contains tokens unrecognized by the model. Provide unmodified heavy and/or light chain sequences."
            )
        aa_idxs = aa_idxs.to(self.device)
        # This makes the expected mask because of
        # test_common.py::test_compare_mask_tensors.
        mask = aa_mask_tensor_of(aa_str)
        mask = mask.to(self.device)
        return aa_idxs.unsqueeze(0), mask.unsqueeze(0)

    def represent_aa_str(self, aa_sequence):
        """Call the forward method of the model on the provided heavy, light pair of AA
        sequences."""
        if isinstance(aa_sequence, str) or len(aa_sequence) != 2:
            raise ValueError(
                "aa_sequence must be a pair of strings, with the first element being the heavy chain sequence and the second element being the light chain sequence."
            )
        inputs = self.prepare_aa_str(*aa_sequence)
        with torch.no_grad():
            return self.represent(*inputs).squeeze(0)

    def selection_factors_of_aa_str(self, aa_sequence: Tuple[str, str]) -> Tensor:
        """Do the forward method then exponentiation without gradients from an amino
        acid string.

        Insertion of model tokens will be done automatically.

        Args:
            aa_str: A heavy, light chain pair of amino acid sequences.

        Returns:
            A tuple of numpy arrays of the same length as the input strings representing
            the level of selection for each amino acid at each site.
        """
        if isinstance(aa_sequence, str) or len(aa_sequence) != 2:
            raise ValueError(
                "aa_sequence must be a pair of strings, with the first element being the heavy chain sequence and the second element being the light chain sequence."
            )
        idx_seq, mask = self.prepare_aa_str(*aa_sequence)
        with torch.no_grad():
            result = torch.exp(self.forward(idx_seq, mask)).squeeze(0)
        return split_heavy_light_model_outputs(result.cpu(), idx_seq.squeeze(0).cpu())

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        """Build a binary log selection matrix from an index-encoded parent sequence.

        Because we're predicting log of the selection factor, we don't use an
        activation function after the transformer.

        Args:
            amino_acid_indices: A tensor of shape (B, L) containing the indices of parent AA sequences.
            mask: A tensor of shape (B, L) representing the mask of valid amino acid sites.

        Returns:
            A tensor of shape (B, L, out_features) representing the log level
            of selection for each possible amino acid at each site.
        """
        result = self.predict(self.represent(amino_acid_indices, mask))
        if self.hyperparameters["output_dim"] >= 20:
            # To match the paper, we set wildtype aa selection factors to 1,
            # so that synonymous codon probabilities are not modified.
            result = zap_predictions_along_diagonal(
                result, amino_acid_indices, fill=0.0
            )
        return result


class TransformerBinarySelectionModelLinAct(AbstractBinarySelectionModel):
    def __init__(
        self,
        nhead: int,
        d_model_per_head: int,
        dim_feedforward: int,
        layer_count: int,
        dropout_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        # Note that d_model has to be divisible by nhead, so we make that
        # automatic here.
        self.d_model_per_head = d_model_per_head
        self.d_model = d_model_per_head * nhead
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.pos_encoder = PositionalEncoding(self.d_model, dropout_prob)
        self.amino_acid_embedding = nn.Embedding(self.known_token_count, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(self.d_model, self.output_dim)
        self.init_weights()

    @property
    def hyperparameters(self):
        return super().hyperparameters | {
            "nhead": self.nhead,
            "d_model_per_head": self.d_model_per_head,
            "dim_feedforward": self.dim_feedforward,
            "layer_count": self.encoder.num_layers,
            "dropout_prob": self.pos_encoder.dropout.p,
        }

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def represent(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        """Represent an index-encoded parent sequence in the model's embedding space.

        Args:
            amino_acid_indices: A tensor of shape (B, L) containing the
                indices of parent AA sequences.
            mask: A tensor of shape (B, L) representing the mask of valid
                amino acid sites.

        Returns:
            The embedded parent sequences, in a tensor of shape (B, L, E),
            where E is the dimensionality of the embedding space.
        """
        # Multiply by sqrt(d_model) to match the transformer paper.
        embedded_amino_acids = self.amino_acid_embedding(
            amino_acid_indices
        ) * math.sqrt(self.d_model)
        # Have to do the permutation because the positional encoding expects the
        # sequence length to be the first dimension.
        embedded_amino_acids = self.pos_encoder(
            embedded_amino_acids.permute(1, 0, 2)
        ).permute(1, 0, 2)

        # To learn about src_key_padding_mask, see https://stackoverflow.com/q/62170439
        return self.encoder(embedded_amino_acids, src_key_padding_mask=~mask)

    def predict(self, representation: Tensor) -> Tensor:
        """Predict selection from the model embedding of a parent sequence.

        Args:
            representation: A tensor of shape (B, L, E) representing the
                embedded parent sequences.
        Returns:
            A tensor of shape (B, L, out_features) representing the log level
            of selection for each amino acid site.
        """
        return self.linear(representation).squeeze(-1)


def wiggle(x, beta):
    """A function that when we exp it gives us a function that slopes to 0 at -inf and
    grows sub-linearly as x increases.

    See https://github.com/matsengrp/netam/pull/5#issuecomment-1906665475 for a
    plot.
    """
    return beta * torch.where(x < 1, x - 1, torch.log(x))


class TransformerBinarySelectionModelWiggleAct(TransformerBinarySelectionModelLinAct):
    """Here the beta parameter is fixed at 0.3."""

    def predict(self, representation: Tensor):
        return wiggle(super().predict(representation), 0.3)


class TransformerBinarySelectionModelTrainableWiggleAct(
    TransformerBinarySelectionModelLinAct
):
    """This version of the model has a trainable parameter that controls the beta in the
    wiggle function.

    It didn't work any better so I'm not using it for now.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the logit of beta to logit(0.3)
        init_beta = 0.3
        init_logit_beta = math.log(init_beta / (1 - init_beta))
        self.logit_beta = nn.Parameter(
            torch.tensor([init_logit_beta], dtype=torch.float32)
        )

    def predict(self, representation: Tensor):
        # Apply sigmoid to transform logit_beta back to the range (0, 1)
        beta = torch.sigmoid(self.logit_beta)
        return wiggle(super().predict(representation), beta)


def reverse_padded_tensors(padded_tensors, padding_mask, padding_value, reversed_dim=1):
    """Reverse the valid values in provided padded_tensors along the specified
    dimension, keeping padding in the same place. For example, if the input is left-
    aligned amino acid sequences and masks, move the padding to the right of the
    reversed sequence. Equivalent to right-aligning the sequences then reversing them. A
    sequence `123456XXXXX` becomes `654321XXXXX`.

    The original padding mask remains valid for the returned tensor.

    Args:
        padded_tensors: (B, L) tensor of amino acid indices
        padding_mask: (B, L) tensor of masks, with True indicating valid values, and False indicating padding values.
        padding_value: The value to fill returned tensor where padding_mask is False.
        reversed_dim: The dimension along which to reverse the tensor. When input is a batch of sequences to be reversed, the default value of 1 is the correct choice.
    Returns:
        The reversed tensor, with the same shape as padded_tensors, and with padding still specified by padding_mask.
    """
    reversed_indices = torch.full_like(padded_tensors, padding_value)
    reversed_indices[padding_mask] = padded_tensors.flip(reversed_dim)[
        padding_mask.flip(reversed_dim)
    ]
    return reversed_indices


class BidirectionalTransformerBinarySelectionModel(AbstractBinarySelectionModel):
    def __init__(
        self,
        nhead: int,
        d_model_per_head: int,
        dim_feedforward: int,
        layer_count: int,
        dropout_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.d_model_per_head = d_model_per_head
        self.d_model = d_model_per_head * nhead
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        # Forward direction components
        self.forward_pos_encoder = PositionalEncoding(self.d_model, dropout_prob)
        self.forward_amino_acid_embedding = nn.Embedding(
            self.known_token_count, self.d_model
        )
        self.forward_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.forward_encoder = nn.TransformerEncoder(
            self.forward_encoder_layer, layer_count
        )

        # Reverse direction components
        self.reverse_pos_encoder = PositionalEncoding(self.d_model, dropout_prob)
        self.reverse_amino_acid_embedding = nn.Embedding(
            self.known_token_count, self.d_model
        )
        self.reverse_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.reverse_encoder = nn.TransformerEncoder(
            self.reverse_encoder_layer, layer_count
        )

        # Output layers
        self.combine_features = nn.Linear(2 * self.d_model, self.d_model)
        self.output = nn.Linear(self.d_model, self.output_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.combine_features.bias.data.zero_()
        self.combine_features.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def single_direction_represent_sequence(
        self,
        indices: Tensor,
        mask: Tensor,
        embedding: nn.Embedding,
        pos_encoder: PositionalEncoding,
        encoder: nn.TransformerEncoder,
    ) -> Tensor:
        """Process sequence through one direction of the model."""
        embedded = embedding(indices) * math.sqrt(self.d_model)
        embedded = pos_encoder(embedded.permute(1, 0, 2)).permute(1, 0, 2)
        return encoder(embedded, src_key_padding_mask=~mask)

    def represent(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        # This is okay, as long as there are no masked ambiguities in the
        # interior of the sequence... Otherwise it should also work for paired seqs.

        # Forward direction - normal processing
        forward_repr = self.single_direction_represent_sequence(
            amino_acid_indices,
            mask,
            self.forward_amino_acid_embedding,
            self.forward_pos_encoder,
            self.forward_encoder,
        )

        # Reverse direction - flip sequences and masks
        reversed_indices = reverse_padded_tensors(
            amino_acid_indices, mask, AA_PADDING_TOKEN
        )

        reverse_repr = self.single_direction_represent_sequence(
            reversed_indices,
            mask,
            self.reverse_amino_acid_embedding,
            self.reverse_pos_encoder,
            self.reverse_encoder,
        )

        # un-reverse to align with forward representation
        aligned_reverse_repr = reverse_padded_tensors(reverse_repr, mask, 0.0)

        # Combine features
        combined = torch.cat([forward_repr, aligned_reverse_repr], dim=-1)
        return self.combine_features(combined)

    def predict(self, representation: Tensor) -> Tensor:
        # Output layer
        return self.output(representation).squeeze(-1)

    @property
    def hyperparameters(self):
        return super().hyperparameters | {
            "nhead": self.nhead,
            "d_model_per_head": self.d_model_per_head,
            "dim_feedforward": self.dim_feedforward,
            "layer_count": self.forward_encoder.num_layers,
            "dropout_prob": self.forward_pos_encoder.dropout.p,
        }


class BidirectionalTransformerBinarySelectionModelWiggleAct(
    BidirectionalTransformerBinarySelectionModel
):
    """Here the beta parameter is fixed at 0.3."""

    def predict(self, representation: Tensor):
        return wiggle(super().predict(representation), 0.3)


class SingleValueBinarySelectionModel(AbstractBinarySelectionModel):
    """A one parameter selection model as a baseline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.single_value = nn.Parameter(torch.tensor(0.0))

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        """Build a binary log selection matrix from an index-encoded parent sequence."""
        if self.output_dim == 1:
            return self.single_value.expand(amino_acid_indices.shape)
        else:
            return self.single_value.expand(
                amino_acid_indices.shape + (self.output_dim,)
            )


class ParentIndependentBinarySelectionModel(AbstractBinarySelectionModel):
    """A parent-independent (position-specific) selection model.

    This model learns a fixed tensor of log selection factors that depends only on
    position, not on the parent sequence. It serves as a neural network analog to
    phylogenetic methods like Bloom & Neher (2023) that estimate position-specific
    fitness effects.

    The learned parameters represent log selection factors that, when exponentiated and
    multiplied by neutral rates, give observed mutation rates.
    """

    def __init__(self, wildtype_sequence: str = None, **kwargs):
        super().__init__(**kwargs)

        # Initialize wildtype sequence and derive max_seq_len from it if provided
        if wildtype_sequence is not None:
            self.wildtype_sequence = wildtype_sequence
            self.max_seq_len = len(wildtype_sequence)
            # Convert wildtype sequence to amino acid indices
            self.wildtype_aa_idxs = aa_idx_tensor_of_str_ambig(wildtype_sequence)
        else:
            # Fallback to default max_seq_len if no wildtype sequence provided
            self.wildtype_sequence = None
            self.max_seq_len = 512
            self.wildtype_aa_idxs = None

        # Initialize the log selection factors
        if self.output_dim == 1:
            self.log_selection_factors = nn.Parameter(torch.zeros(self.max_seq_len))
        else:
            self.log_selection_factors = nn.Parameter(
                torch.zeros(self.max_seq_len, self.output_dim)
            )

    @property
    def hyperparameters(self):
        hyperparams = super().hyperparameters | {
            "max_seq_len": self.max_seq_len,
        }
        if self.wildtype_sequence is not None:
            hyperparams["wildtype_sequence"] = self.wildtype_sequence
        return hyperparams

    def forward(self, amino_acid_indices: Tensor, mask: Tensor) -> Tensor:
        """Return position-specific log selection factors.

        Since this is a parent-independent model, the amino_acid_indices parameter
        is ignored - we only use it to determine batch size and sequence length.
        The selection factors depend only on position, not on the parent sequence.
        Masked positions (where mask is False) will have their log selection factors
        set to 0, which becomes a selection factor of 1 after exponentiation.

        Args:
            amino_acid_indices: A tensor of shape (B, L) - used only for shape.
            mask: A tensor of shape (B, L) representing the mask of valid amino acid sites.
                  True for valid positions, False for positions to be masked.

        Returns:
            A tensor of shape (B, L) or (B, L, output_dim) with log selection factors.
        """
        batch_size, seq_len = amino_acid_indices.shape

        # Get the relevant slice of our position-specific parameters
        position_factors = self.log_selection_factors[:seq_len]

        # Expand to match the required batch and output shape
        if self.output_dim == 1:
            result = position_factors.expand(batch_size, seq_len)
            # Apply masking: multiplicative masking in log space (consistent with other models)
            result = result * mask
        else:
            # Create a proper copy instead of a view to avoid in-place operation issues
            result = (
                position_factors.unsqueeze(0)
                .expand(batch_size, seq_len, self.output_dim)
                .clone()
            )

            # Apply zap_predictions_along_diagonal for multi-dimensional output
            if self.output_dim >= 20 and self.wildtype_aa_idxs is not None:
                # Use the wildtype sequence indices for all sequences in the batch
                # Expand wildtype indices to match batch size
                wt_idxs_batch = (
                    self.wildtype_aa_idxs[:seq_len].unsqueeze(0).expand(batch_size, -1)
                )
                # Set wildtype aa selection factors to 0 (which becomes 1 after exp)
                result = zap_predictions_along_diagonal(result, wt_idxs_batch, fill=0.0)

            # Apply masking: expand mask to match output dimensions
            result = result * mask.unsqueeze(-1)

        return result


class HitClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reinitialize_weights()

    @property
    def hyperparameters(self):
        return {}

    def forward(
        self, parent_codon_idxs: torch.Tensor, uncorrected_codon_probs: torch.Tensor
    ):
        """Forward function takes a tensor of target codon distributions, for each
        observed parent codon, and adjusts the distributions according to the hit class
        corrections.

        Inputs should be in unflattened form, with codons represented in shape (4, 4,
        4).
        """
        result = self.apply_multihit_correction(
            parent_codon_idxs, uncorrected_codon_probs
        )
        # clamp only above to avoid summing a bunch of small fake values when
        # computing wild type prob
        unnormalized_corrected_probs = clamp_probability_above_only(result)
        # Recompute parent codon probability
        result = molevol.set_parent_codon_prob(
            molevol.flatten_codons(unnormalized_corrected_probs),
            flatten_codon_idxs(parent_codon_idxs),
        )
        # Clamp again to ensure parent codon probabilities are valid.
        result = clamp_probability(result)
        return molevol.unflatten_codons(result)

    def apply_multihit_correction(
        self, parent_codon_idxs: torch.Tensor, uncorrected_codon_probs: torch.Tensor
    ):
        """Apply the correction to the uncorrected codon probabilities.

        Unlike `forward` this does not clamp or recompute parent codon probability.
        Otherwise, it is identical to `forward`.
        """
        return apply_multihit_correction(
            parent_codon_idxs, uncorrected_codon_probs, self.values
        )

    def reinitialize_weights(self, parameters=(0.0, 0.0, 0.0)):
        self.values = nn.Parameter(torch.tensor(parameters))

    def to_weights(self):
        return tuple(self.values.detach().numpy())

    @classmethod
    def from_weights(cls, weights):
        assert len(weights) == 3
        model = cls()
        model.reinitialize_weights(weights)
        model.eval()
        return model
