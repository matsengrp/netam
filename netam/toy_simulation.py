"""
These are functions for simulating amino acid mutations in a protein sequence.

So, this is not for simulating mutation-selection processes.

It corresponds to the inference happning in toy_dnsm.py.
"""

import random

import pandas as pd
from tqdm import tqdm

from epam.sequences import AA_STR_SORTED


def mimic_mutations(sequence_mutator_fn, parents, sub_counts):
    """
    Mimics mutations for a series of parent sequences.

    Parameters
    ----------
    sequence_mutator_fn : function
        Function that takes a string sequence and an integer, returns a mutated sequence with that many mutations.
    parents : pd.Series
        Series containing parent sequences as strings.
    sub_counts : pd.Series
        Series containing the number of substitutions for each parent sequence.

    Returns
    -------
    pd.Series
        Series containing mutated sequences as strings.
    """

    mutated_sequences = []

    for seq, sub_count in tqdm(
        zip(parents, sub_counts), total=len(parents), desc="Mutating sequences"
    ):
        mutated_seq = sequence_mutator_fn(seq, sub_count)
        mutated_sequences.append(mutated_seq)

    return pd.Series(mutated_sequences)


def general_mutator(aa_seq, sub_count, mut_criterion):
    """
    General function to mutate an amino acid sequence based on a criterion function.
    The function first identifies positions in the sequence that satisfy the criterion
    specified by `mut_criterion`. If the number of such positions is less than or equal
    to the number of mutations needed (`sub_count`), then mutations are made at those positions.
    If `sub_count` is greater than the number of positions satisfying the criterion, the function
    mutates all those positions and then randomly selects additional positions to reach `sub_count`
    total mutations. All mutations change the amino acid to a randomly selected new amino acid,
    avoiding a mutation to the same type.

    Parameters
    ----------
    aa_seq : str
        Original amino acid sequence.
    sub_count : int
        Number of substitutions to make.
    mut_criterion : function
        Function that takes a sequence and a position, returns True if position should be mutated.

    Returns
    -------
    str
        Mutated amino acid sequence.
    """

    def draw_new_aa_for_pos(pos):
        return random.choice([aa for aa in AA_STR_SORTED if aa != aa_seq_list[pos]])

    aa_seq_list = list(aa_seq)

    # find all positions that satisfy the mutation criterion
    mut_positions = [
        pos for pos, aa in enumerate(aa_seq_list) if mut_criterion(aa_seq, pos)
    ]

    # if fewer criterion-satisfying positions than required mutations, randomly add more
    if len(mut_positions) < sub_count:
        extra_positions = random.choices(
            [pos for pos in range(len(aa_seq_list)) if pos not in mut_positions],
            k=sub_count - len(mut_positions),
        )
        mut_positions += extra_positions

    # if more criterion-satisfying positions than required mutations, randomly remove some
    elif len(mut_positions) > sub_count:
        mut_positions = random.sample(mut_positions, sub_count)

    # perform mutations
    for pos in mut_positions:
        aa_seq_list[pos] = draw_new_aa_for_pos(pos)

    return "".join(aa_seq_list)


# Criterion functions
def tyrosine_mut_criterion(aa_seq, pos):
    return aa_seq[pos] == "Y"


def hydrophobic_mut_criterion(aa_seq, pos):
    hydrophobic_aa = set("AILMFVWY")
    return aa_seq[pos] in hydrophobic_aa


def hydrophobic_neighbor_mut_criterion(aa_seq, pos):
    """
    Criterion function that returns True if either amino acid at immediate
    neighbors are hydrophobic.
    """

    hydrophobic_aa = set("AILMFVWY")

    positions_to_check = []
    if pos > 0:
        positions_to_check.append(pos - 1)
    if pos < len(aa_seq) - 1:
        positions_to_check.append(pos + 1)

    return any(aa_seq[i] in hydrophobic_aa for i in positions_to_check)


[tyrosine_mutator, hydrophobic_mutator, hydrophobic_neighbor_mutator] = [
    lambda aa_seq, sub_count, crit=crit: general_mutator(aa_seq, sub_count, crit)
    for crit in [
        tyrosine_mut_criterion,
        hydrophobic_mut_criterion,
        hydrophobic_neighbor_mut_criterion,
    ]
]
