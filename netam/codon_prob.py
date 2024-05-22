from framework import Burrito, trimmed_shm_model_outputs_of_crepe
import torch
from tqdm import tqdm

from epam.molevol import (
    reshape_for_codons,
    build_mutation_matrices,
    codon_probs_of_mutation_matrices,
)
from epam.torch_common import optimize_branch_length

def codon_probs_of_parent_scaled_rates_and_sub_probs(
    parent_idxs, scaled_rates, sub_probs
):
    """
    Compute the probabilities of mutating to various codons for a parent sequence.

    This uses the same machinery as we use for fitting the DNSM, but we stay on
    the codon level rather than moving to syn/nonsyn changes.
    """
    # This is from `aaprobs_of_parent_scaled_rates_and_sub_probs`:
    mut_probs = 1.0 - torch.exp(-scaled_rates)
    parent_codon_idxs = reshape_for_codons(parent_idxs)
    codon_mut_probs = reshape_for_codons(mut_probs)
    codon_sub_probs = reshape_for_codons(sub_probs)

    # This is from `aaprob_of_mut_and_sub`:
    mut_matrices = build_mutation_matrices(
        parent_codon_idxs, codon_mut_probs, codon_sub_probs
    )
    codon_probs = codon_probs_of_mutation_matrices(mut_matrices)

    return codon_probs

def _trim_seqs_to_codon_boundary_and_max_len(seqs, site_count):
    """Assumes that all sequences have the same length"""
    max_len = site_count - site_count % 3
    return [seq[: min(len(seq) - len(seq) % 3, max_len)] for seq in seqs]

def _prepare_pcp_df(pcp_df, crepe, site_count):
    """
    Trim the sequences to codon boundaries and add the rates and substitution probabilities.
    """
    pcp_df["parent"] = _trim_seqs_to_codon_boundary_and_max_len(
        pcp_df["parent"], site_count
    )
    pcp_df["child"] = _trim_seqs_to_codon_boundary_and_max_len(
        pcp_df["child"], site_count
    )
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]].reset_index(drop=True)
    ratess, cspss = trimmed_shm_model_outputs_of_crepe(
        crepe, pcp_df["parent"]
    )
    pcp_df["rates"] = ratess
    pcp_df["subs_probs"] = cspss
    return pcp_df

class CodonProbBurrito(Burrito):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        model,
        site_count,
        crepe,
        **kwargs,
    ):
        train_dataset = _prepare_pcp_df(train_dataset, crepe, site_count)
        assert train_dataset["parent"].apply(len).max() <= site_count - site_count % 3
        val_dataset = _prepare_pcp_df(val_dataset, crepe, site_count)
        super().__init__(
            train_dataset,
            val_dataset,
            model,
            **kwargs,
        )
    
    # These are from RSSHMBurrito, as a start
    def _find_optimal_branch_length(
        self,
        encoded_parent,
        mask,
        mutation_indicator,
        wt_base_modifier,
        starting_branch_length,
        **optimization_kwargs,
    ):
        if torch.sum(mutation_indicator) == 0:
            return 0.0

        rates, _ = self.model(
            encoded_parent.unsqueeze(0),
            mask.unsqueeze(0),
            wt_base_modifier.unsqueeze(0),
        )

        rates = rates.squeeze().double()
        mutation_indicator_masked = mutation_indicator[mask].double()

        # From issue: "Just to get things going, let's have our log probability just be the probability of getting a codon mutation vs no."
        def log_pcp_probability(log_branch_length):
            branch_length = torch.exp(log_branch_length)
            mut_prob = 1 - torch.exp(-rates * branch_length)
            mut_prob_masked = mut_prob[mask]
            rate_loss = self.bce_loss(mut_prob_masked, mutation_indicator_masked)
            return -rate_loss

        return optimize_branch_length(
            log_pcp_probability,
            starting_branch_length.double().item(),
            **optimization_kwargs,
        )

    def find_optimal_branch_lengths(self, dataset, **optimization_kwargs):
        optimal_lengths = []

        self.model.eval()
        self.model.freeze()

        for (
            encoded_parent,
            mask,
            mutation_indicator,
            wt_base_modifier,
            starting_branch_length,
        ) in tqdm(
            zip(
                dataset.encoded_parents,
                dataset.masks,
                dataset.mutation_indicators,
                dataset.wt_base_modifier,
                dataset.branch_lengths,
            ),
            total=len(dataset.encoded_parents),
            desc="Finding optimal branch lengths",
        ):
            optimal_lengths.append(
                self._find_optimal_branch_length(
                    encoded_parent,
                    mask,
                    mutation_indicator,
                    wt_base_modifier,
                    starting_branch_length,
                    **optimization_kwargs,
                )
            )

        self.model.unfreeze()

        return torch.tensor(optimal_lengths)

class CodonProbDataset:
    pass

