"""Code for generating observed vs expected plots."""

import numpy as np
import pandas as pd
import bisect
from netam.sequences import (
    AA_STR_SORTED,
    translate_sequence,
)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def check_for_out_of_range(df):
    """Check for probabilities out of range in a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with column 'prob'.
    """
    out_of_range = df[(df["prob"] < 0) | (df["prob"] > 1)]
    if len(out_of_range) > 0:
        print(
            f"Warning: {len(out_of_range)} rows of {len(df)} have probabilities out of range."
        )
        print("Here's a sample:")
        print(out_of_range.head())


def annotate_sites_df(
    df,
    pcp_df,
    numbering_dict=None,
    add_codons_aas=False,
):
    """Add annotations to a per-site DataFrame to indicate position of each site,
    whether each site is in a CDR, and optionally parent/child codon and amino acid
    information. The input DataFrame describes a site in each row is expected to have
    the 'pcp_index' column, indicating the index of the PCP the site belongs to.

    Parameters:
    df (pd.DataFrame): site mutabilities DataFrame.
    pcp_df (pd.DataFrame): PCP file of the dataset.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    add_codons_aas (bool): whether to add codon and amino acid information from parent/child sequences.

    Returns:
    output_df (pd.DataFrame): dataframe with additional columns 'site' and 'is_cdr'.
                              If add_codons_aas=True, also adds columns 'parent_codon', 'parent_aa',
                              'child_codon', and 'child_aa'.
                              Note that if numbering_dict is provided and there are clonal families with missing
                              ANARCI numberings, the associated sites (rows) will be excluded.
    """
    sites_col = []
    is_cdr_col = []

    if add_codons_aas:
        parent_codon_col = []
        parent_aa_col = []
        child_codon_col = []
        child_aa_col = []

    pcp_groups = df.groupby("pcp_index")
    for pcp_index in df["pcp_index"].drop_duplicates():
        pcp_row = pcp_df.loc[pcp_index]

        group_df = pcp_groups.get_group(pcp_index)
        nsites = group_df.shape[0]

        # Assert that sequence length is divisible by 3 and check the number of sites in substitution df vs nuclotide sequence
        assert (
            len(pcp_row["parent"]) % 3 == 0
        ), f"Parent sequence length ({len(pcp_row['parent'])}) is not divisible by 3"
        assert (
            nsites == len(pcp_row["parent"]) // 3
        ), f"Number of sites ({nsites}) does not match sequence length ({len(pcp_row['parent']) // 3})"

        if numbering_dict is None:
            sites_col.append(np.arange(nsites))
        else:
            nbkey = tuple(pcp_row[["sample_id", "family"]])
            if nbkey in numbering_dict:
                sites_col.append(numbering_dict[nbkey])
            else:
                # Assign sites as "None", marking them for exclusion from output.
                sites_col.append(["None"] * nsites)

        is_cdr_col.append(pcp_sites_cdr_annotation(pcp_row))

        if add_codons_aas:
            parent_seq = pcp_row["parent"]
            child_seq = pcp_row["child"]
            parent_aa_seq = translate_sequence(parent_seq)
            child_aa_seq = translate_sequence(child_seq)

            parent_codons = [parent_seq[(3 * i) : (3 * i) + 3] for i in range(nsites)]
            parent_aas = list(parent_aa_seq)
            child_codons = [child_seq[(3 * i) : (3 * i) + 3] for i in range(nsites)]
            child_aas = list(child_aa_seq)

            parent_codon_col.append(parent_codons)
            parent_aa_col.append(parent_aas)
            child_codon_col.append(child_codons)
            child_aa_col.append(child_aas)

    df["site"] = np.concatenate(sites_col)
    df["is_cdr"] = np.concatenate(is_cdr_col)

    if add_codons_aas:
        df["parent_codon"] = np.concatenate(parent_codon_col)
        df["parent_aa"] = np.concatenate(parent_aa_col)
        df["child_codon"] = np.concatenate(child_codon_col)
        df["child_aa"] = np.concatenate(child_aa_col)

    if numbering_dict is None:
        return df
    else:
        return df[df["site"] != "None"]


def plot_observed_vs_expected(
    df,
    counts_ax,
    oe_ax,
    diff_ax,
    logprobs=True,
    binning=None,
    counts_color="#B3CDE3",
    pcurve_color="#D95F02",
    model_color="#0072B2",
    model_name="Expected",
    logy=False,
    normalize=False,
):
    """
    Draws a figure with up to 3 panels showing:
    counts of sites in bins of amino acid substitution probability,
    observed vs expected number of mutations in bins of amino acid substitution probability,
    and per bin differences between observed and expected.
    The expected number of mutations is computed as the total probability
    of the sites that fall in that mutability bin.
    The input dataframe requires two columns: 'prob'
    (site mutability -- may be at level of nucleotide, or codon, or amino acid, etc.)
    and 'mutation' (True/False whether the site has an observed mutation or not).
    Each dataframe row corresponds to a site in a specific sequence.
    Thus, the total number of rows is the total number of sites from
    all sequences in the dataset.

    Parameters:
    df (pd.DataFrame): dataframe of site mutabilities.
    counts_ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    oe_ax (fig.ax): figure axis for plotting observed vs expected number of mutations. If None, plot is not drawn.
    diff_ax (fig.ax): figure axis for ploting observed vs expected differences. If None, plot is not drawn.
    logprobs (bool): whether to plot log-probabilities (True) or plot probabilities (False).
    binning (list): list of bin boundaries (i.e. n+1 boundaries for n bins). If None, a default binning is used.
    counts_color (str): color for the counts of sites plot.
    pcurve_color (str): color for the probability curve in the counts of sites plot.
    model_color (str): color for the plot of expected number of mutations.
    model_name (str): legend label for the plot of expected number of mutations.
    logy (bool): whether to show y-axis in log-scale.
    normalize (bool): whether to scale the area of the expected mutations distribution to match the observed mutations distribution.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.
    counts_twinx_ax (fig.ax): handle to the probability y-axis (right side) of the counts plot, if drawn.

    """
    check_for_out_of_range(df)
    model_probs = df["prob"].to_numpy()

    # set default binning if None specified
    if binning is None:
        if logprobs:
            min_logprob = 1.05 * np.log10(model_probs).min()
            binning = np.linspace(min_logprob, 0, 101)
        else:
            max_prob = min(1, 1.05 * model_probs.max())
            binning = np.linspace(0, max_prob, 101)

    # compute expectation
    bin_index_col = []
    for p in model_probs:
        if logprobs:
            index = bisect.bisect(binning, np.log10(p)) - 1
        else:
            index = bisect.bisect(binning, p) - 1
        bin_index_col.append(index)
    df["bin_index"] = bin_index_col

    expected = []
    exp_err = []
    for i in range(len(binning) - 1):
        binprobs = df[df["bin_index"] == i]["prob"].to_numpy()
        expected.append(np.sum(binprobs))
        exp_err.append(np.sqrt(np.sum(binprobs * (1 - binprobs))))
    expected = np.array(expected)
    exp_err = np.array(exp_err)

    # count observed mutations
    if logprobs:
        obs_probs = np.log10(df[df["mutation"] > 0]["prob"].to_numpy())
        xlabel = r"$\log_{10}$(amino acid substitution probability)"
    else:
        obs_probs = df[df["mutation"] > 0]["prob"].to_numpy()
        xlabel = "amino acid substitution probability"
    observed = np.histogram(obs_probs, binning)[0]

    # normalize total expected to equal total observed
    if normalize:
        fnorm = np.sum(observed) / np.sum(expected)
        expected = fnorm * expected
        exp_err = fnorm * exp_err

    # compute overlap metric
    intersect = np.sum(np.minimum(observed, expected))
    denom = 0.5 * (np.sum(observed) + np.sum(expected))
    overlap = intersect / denom

    # compute residual metric
    diff = observed - expected
    residual = np.sqrt(np.sum(diff * diff)) / np.sum(expected)

    # midpoints of each bin
    xvals = [0.5 * (binning[i] + binning[i + 1]) for i in range(len(binning) - 1)]

    # bin widths
    binw = [(binning[i + 1] - binning[i]) for i in range(len(binning) - 1)]

    # plot site counts
    counts_twinx_ax = None
    if counts_ax is not None:
        if logprobs:
            hist_data = np.log10(model_probs)
        else:
            hist_data = model_probs
        counts_ax.hist(hist_data, bins=binning, color=counts_color)
        if (oe_ax is None) and (diff_ax is None):
            counts_ax.tick_params(axis="x", labelsize=16)
            counts_ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        counts_ax.tick_params(axis="y", labelsize=16)
        counts_ax.set_ylabel("number of sites", fontsize=20, labelpad=10)
        counts_ax.grid()

        if logy:
            counts_ax.set_yscale("log")

        if logprobs:
            yvals = np.power(10, xvals)
        else:
            yvals = xvals

        counts_twinx_ax = counts_ax.twinx()
        counts_twinx_ax.plot(xvals, yvals, color=pcurve_color)
        counts_twinx_ax.tick_params(axis="y", labelcolor=pcurve_color, labelsize=16)
        counts_twinx_ax.set_ylabel("probability", fontsize=20, labelpad=10)
        counts_twinx_ax.set_ylim(0, 1)

    # plot observed vs expected number of mutations
    if oe_ax is not None:
        oe_ax.bar(
            xvals,
            expected,
            width=binw,
            facecolor="white",
            edgecolor=model_color,
            label=model_name,
        )
        oe_ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
            label="Observed",
        )
        if diff_ax is None:
            oe_ax.tick_params(axis="x", labelsize=16)
            oe_ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        oe_ax.tick_params(axis="y", labelsize=16)
        oe_ax.set_ylabel("number of mutations", fontsize=20, labelpad=10)

        # For some reason, regardless of draw order, legend labels are always ordered:
        #   Observed
        #   Model
        # Force reverse the order.
        leg_handles, leg_labels = oe_ax.get_legend_handles_labels()
        oe_ax.legend(leg_handles[::-1], leg_labels[::-1], fontsize=15)

        if logy:
            oe_ax.set_yscale("log")

        boxes0 = [
            Rectangle(
                (binning[ibin], expected[ibin] - exp_err[ibin]),
                binning[ibin + 1] - binning[ibin],
                2 * exp_err[ibin],
            )
            for ibin in range(len(exp_err))
        ]
        pc0 = PatchCollection(
            boxes0, facecolor="none", edgecolor="grey", linewidth=0, hatch="//////"
        )
        oe_ax.add_collection(pc0)

    # plot observed vs expected difference
    if diff_ax is not None:
        diff_ax.plot(
            xvals,
            [yo - ye for yo, ye in zip(observed, expected)],
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
        )
        diff_ax.axhline(y=0, color="k", linestyle="--")
        diff_ax.tick_params(axis="x", labelsize=16)
        diff_ax.set_xlabel(xlabel, fontsize=20, labelpad=10)
        diff_ax.tick_params(axis="y", labelsize=16)
        diff_ax.set_ylabel("Obs - Exp", fontsize=20, labelpad=10)

        boxes1 = [
            Rectangle(
                (binning[ibin], -exp_err[ibin]),
                binning[ibin + 1] - binning[ibin],
                2 * exp_err[ibin],
            )
            for ibin in range(len(exp_err))
        ]
        pc1 = PatchCollection(
            boxes1, facecolor="none", edgecolor="grey", linewidth=0, hatch="//////"
        )
        diff_ax.add_collection(pc1)

    return {
        "overlap": overlap,
        "residual": residual,
        "counts_twinx_ax": counts_twinx_ax,
    }


def plot_sites_observed_vs_expected(
    df,
    ax,
    numbering_dict=None,
    fwr_color="#0072B2",
    cdr_color="#E69F00",
    logy=False,
):
    """
    Draws a figure of observed vs expected number of mutations at each site position across PCPs in a dataset.
    The input dataframe requires four columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'prob' (site mutability)
    'mutation' (True/False whether the site has an observed mutation or not)
    'is_cdr' (True/False whether the site is in a CDR)
    Each dataframe row corresponds to a site in a specific sequence.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    fwr_color (str): color for the FWR sites.
    cdr_color (str): color for the CDR sites.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.

    """
    check_for_out_of_range(df)

    if numbering_dict is None:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)
    else:
        xvals = numbering_dict[("reference", 0)]
    ixvals = np.arange(len(xvals))

    expected = []
    exp_err = []
    observed = []
    fwr_expected = []
    for site in xvals:
        site_df = df[df["site"] == site]
        site_probs = site_df["prob"].to_numpy()
        expected.append(np.sum(site_probs))
        exp_err.append(np.sqrt(np.sum(site_probs * (1 - site_probs))))
        observed.append(df[(df["mutation"] == 1) & (df["site"] == site)].shape[0])
        site_fwr_probs = site_df[~site_df["is_cdr"]]["prob"].to_numpy()
        fwr_expected.append(np.sum(site_fwr_probs))

    expected = np.array(expected)
    exp_err = np.array(exp_err)
    observed = np.array(observed)

    # compute overlap metric
    intersect = np.sum(np.minimum(observed, expected))
    denom = 0.5 * (np.sum(observed) + np.sum(expected))
    overlap = intersect / denom

    # compute residual metric
    diff = observed - expected
    residual = np.sqrt(np.sum(diff * diff)) / np.sum(expected)

    if ax is not None:
        ax.bar(
            xvals,
            expected,
            width=1,
            facecolor="white",
            edgecolor=cdr_color,
            label="Expected (CDR)",
        )

        ax.bar(
            xvals,
            fwr_expected,
            width=1,
            facecolor="white",
            edgecolor=fwr_color,
            label="Expected (FWR)",
        )

        ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
            label="Observed",
        )

        if df.dtypes["site"] == "object":
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        else:
            ax.tick_params(axis="x", labelsize=16)
        ax.set_xlabel("amino acid sequence position", fontsize=20, labelpad=10)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel("number of substitutions", fontsize=20, labelpad=10)
        ax.margins(x=0.01)

        if logy:
            ax.set_yscale("log")

        ax.legend(fontsize=15)

        boxes0 = [
            Rectangle(
                (ixvals[i] - 0.5, expected[i] - exp_err[i]),
                1,
                2 * exp_err[i],
            )
            for i in range(len(expected))
        ]
        pc0 = PatchCollection(
            boxes0, facecolor="none", edgecolor="grey", linewidth=0, hatch="//////"
        )
        ax.add_collection(pc0)

    return {
        "overlap": overlap,
        "residual": residual,
        "total_obs": np.sum(observed),
        "total_exp": np.sum(expected),
    }


def locate_top_k_substitutions(site_sub_probs, k_sub):
    """Return the top k substitutions predicted for a parent-child pair given
    precalculated site substitution probabilities.

    Parameters:
    site_sub_probs (np.array): Probability of substition at each site for a parent sequence.
    k_sub (int): Number of substitutions observed in PCP.

    Returns:
    pred_sub_sites (np.array): Location of top-k predicted substitutions by model (unordered).
    """
    if k_sub == 0:
        return []

    # np.argpartition returns indices of top k elements in unsorted order
    pred_sub_sites = np.argpartition(site_sub_probs, -k_sub)[-k_sub:]

    assert (
        pred_sub_sites.size == k_sub
    ), "The number of predicted substitution sites does not match the number of actual substitution sites."

    return pred_sub_sites


def get_subs_and_preds_from_mutabilities_df(df, pcp_df):
    """Determines the sites of observed and predicted substitutions of every PCP in a
    dataset, from a site mutabilities DataFrame, which has columns 'pcp_index' (index of
    the PCP that the site belongs to), 'prob' (the mutability probability of the site),
    'mutation' (whether the site has an observed mutation). Predicted substitutions are
    the sites in the top-k of mutability, where k is the number of observed substition
    in the PCP.

    Parameters:
    df (pd.DataFrame): site mutabilities DataFrame.
    pcp_df (pd.DataFrame): PCP file of the dataset.

    Returns tuple with:
    pcp_indices (list): indices to the reference PCP file.
    pcp_sub_locations (list): per-PCP lists of substitution locations (positions along the sequence string).
    top_k_sub_locations (list): per-PCP lists of top-k mutability locations (positions along the sequence string).
    pcp_sample_family_dict (dict): mapping PCP index to (sample_id, family) 2-tuple.
    """
    pcp_indices = list(df["pcp_index"].drop_duplicates())
    pcp_sub_locations = []
    top_k_sub_locations = []
    pcp_sample_family_dict = {}

    for pcp_index in pcp_indices:
        probs = list(df[df["pcp_index"] == pcp_index]["prob"])
        mutations = list(df[df["pcp_index"] == pcp_index]["mutation"])
        pcp_sub_locations.append(list(i for i in range(len(mutations)) if mutations[i]))
        top_k_sub_locations.append(locate_top_k_substitutions(probs, sum(mutations)))
        pcp_sample_family_dict[pcp_index] = tuple(
            pcp_df.loc[pcp_index][["sample_id", "family"]]
        )

    return (pcp_indices, pcp_sub_locations, top_k_sub_locations, pcp_sample_family_dict)


def get_site_substitutions_df(
    subs_and_preds_tuple,
    numbering_dict=None,
):
    """Returns a dataframe that annotates for each site of observed and/or predicted
    substitution: the index of the PCP it belongs to, the site position in the amino
    acid sequence, whether the site has an observed substutition, and whether the site
    is predicted to have a substitution.

    Parameters:
    subs_and_preds_tuple (tuple): 4-tuple of
                                  pcp_indices - list of indices to the reference PCP file,
                                  pcp_sub_locations - list of per-PCP lists of substitution locations,
                                  top_k_sub_locations - list of per-PCP lists of top-k mutability locations,
                                  pcp_sample_family_dict - dictionary mapping PCP index to (sample_id, family) 2-tuple.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns 'pcp_index', 'site', 'obs', 'pred'.
    """
    pcp_indices = subs_and_preds_tuple[0]
    pcp_sub_locations = subs_and_preds_tuple[1]
    top_k_sub_locations = subs_and_preds_tuple[2]
    pcp_sample_family_dict = subs_and_preds_tuple[3]

    pcp_index_col = []
    site_col = []
    obs_col = []
    pred_col = []
    for i in range(len(pcp_indices)):
        obs_pred_sites = np.union1d(top_k_sub_locations[i], pcp_sub_locations[i])

        nbkey = pcp_sample_family_dict[pcp_indices[i]]
        if (numbering_dict is not None) and (nbkey not in numbering_dict):
            continue

        for site in obs_pred_sites:
            pcp_index_col.append(pcp_indices[i])
            if numbering_dict is None:
                site_col.append(site)
            else:
                site_col.append(numbering_dict[nbkey][site])
            obs_col.append(site in pcp_sub_locations[i])
            pred_col.append(site in top_k_sub_locations[i])

    output_df = pd.DataFrame(columns=["pcp_index", "site", "obs", "pred"])
    output_df["pcp_index"] = pcp_index_col
    output_df["site"] = site_col
    output_df["obs"] = obs_col
    output_df["pred"] = pred_col

    return output_df


def plot_sites_observed_vs_top_k_predictions(
    df,
    ax,
    numbering_dict=None,
    correct_color="#009E73",
    correct_label="Correct",
    incorrect_color="#009E73",
    incorrect_label="Incorrect",
    logy=False,
):
    """
    Draws a figure of observed mutations and the top-k predictions across PCPs in a dataset.
    The input dataframe requires three columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'obs' (True/False whether the site has an observed substitution)
    'pred' (True/False whether the site is in the top-k predicted substitutions)
    Each dataframe row corresponds to a site in a specific sequence.
    Only sites that are True in either 'obs' or 'pred' columns are involved in the plotting and calculation.
    Hence, sites in a PCP that have neither an observed substitution nor are predicted in the top-k
    can be excluded from the dataframe.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    correct_color (str): color for the plot of correct predictions.
    correct_label (str): legend label for the plot of correct predictions.
    incorrect_color (str): color for the plot of incorrect predictions.
    incorrect_label (str): legend label for the plot of incorrect predictions.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    overlap (float): area of overlap between observed and expected, divided by the average of the two areas.
    residual (float): square root of the sum of squared bin-by-bin differences between observed and expected, divided by total expected.
    r-precision (float): R-precision of the dataset.

    """
    if numbering_dict is None:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)
    else:
        xvals = numbering_dict[("reference", 0)]

    predicted = []
    observed = []
    correct = []
    for site in xvals:
        site_df = df[df["site"] == site]
        npred = site_df[site_df["pred"]].shape[0]
        predicted.append(npred)
        nobs = site_df[site_df["obs"]].shape[0]
        observed.append(nobs)
        ncorr = site_df[site_df["pred"] & site_df["obs"]].shape[0]
        correct.append(ncorr)
    predicted = np.array(predicted)
    observed = np.array(observed)
    correct = np.array(correct)

    # compute overlap metric
    intersect = np.sum(np.minimum(observed, predicted))
    denom = 0.5 * (np.sum(observed) + np.sum(predicted))
    overlap = intersect / denom

    # compute residual metric
    diff = observed - predicted
    residual = np.sqrt(np.sum(diff * diff)) / np.sum(predicted)

    # compute R-precision
    tmpdf = df[df["obs"]][["pcp_index", "obs", "pred"]].groupby("pcp_index").sum()
    pcp_rprec = tmpdf["pred"].to_numpy() / tmpdf["obs"].to_numpy()
    rprec = sum(pcp_rprec) / len(pcp_rprec)

    if ax is not None:
        ax.bar(
            xvals,
            predicted,
            width=1,
            facecolor="white",
            edgecolor=incorrect_color,
            label=incorrect_label,
        )

        ax.bar(
            xvals,
            correct,
            width=1,
            color=correct_color,
            edgecolor=correct_color,
            label=correct_label,
        )

        ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=4,
            linewidth=0,
            color="#000000",
            label="Observed",
        )

        if df.dtypes["site"] == "object":
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        else:
            ax.tick_params(axis="x", labelsize=16)
        ax.set_xlabel("amino acid sequence position", fontsize=20, labelpad=10)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel("number of substitutions", fontsize=20, labelpad=10)
        ax.margins(x=0.01)

        if logy:
            ax.set_yscale("log")

        ax.legend(fontsize=15)

    return {
        "overlap": overlap,
        "residual": residual,
        "r-precision": rprec,
    }


def get_sub_acc_from_csp_df(df, pcp_df, top_k=1):
    """Determines the sites of observed substitutions and whether the amino acid
    substitution is predicted among the top-k most probable for every PCP in a dataset,
    using a dataframe of CSPs at sites of substitutions. The input DataFrame describes a
    site and conditional substitution probability (CSP) in each row for an amino acid.
    The `site` corresponds to a site of substitution, and there are 20 rows for each
    site to cover all amino acids. Each row is expected to have the `pcp_index` column,
    indicating the index of the PCP the site belongs to. `prob` denotes a CSP and `aa`
    denotes the index in AA_STR_SORTED of the corresponding amino acid. `is_target`
    denotes if the amino acid is the observed substitution target.

    Parameters:
    df (pd.DataFrame): site CSPs DataFrame.
    pcp_df (pd.DataFrame): corresponding PCP DataFrame, to access various annotations.

    Returns tuple with:
    pcp_indices (list): indices to the reference PCP file.
    pcp_sub_locations (list): per-PCP lists of substitution locations (positions along the sequence string).
    pcp_sub_correct (list): per-PCP lists of True/False whether substitution prediction contains the correct amino acid.
    pcp_is_cdr (list): per-PCP lists of True/False whether amino acid site is in a CDR.
    pcp_sample_family_dict (dict): mapping PCP index to (sample_id, family) 2-tuple.
    """
    pcp_indices = []
    pcp_sub_locations = []
    pcp_sub_correct = []
    pcp_is_cdr = []
    pcp_sample_family_dict = {}

    pcp_groups = df.groupby("pcp_index")
    for pcp_index in df["pcp_index"].drop_duplicates():
        pcp_row = pcp_df.loc[pcp_index]
        cdr_anno = pcp_sites_cdr_annotation(pcp_row)

        group_df = pcp_groups.get_group(pcp_index)

        pcp_indices.append(pcp_index)

        sub_locations = group_df["site"].drop_duplicates().tolist()
        pcp_sub_locations.append(sub_locations)

        sites_correct = []
        sites_is_cdr = []
        site_groups = group_df.groupby("site")
        for site in sub_locations:
            site_df = site_groups.get_group(site)
            csp = site_df["prob"].to_numpy()
            is_target = site_df["is_target"].to_numpy()

            csp_sorted_indices = csp.argsort()[::-1]
            sites_correct.append(True in is_target[csp_sorted_indices[:top_k]])
            sites_is_cdr.append(cdr_anno[site])

        pcp_sub_correct.append(sites_correct)
        pcp_is_cdr.append(sites_is_cdr)

        pcp_sample_family_dict[pcp_index] = tuple(pcp_row[["sample_id", "family"]])

    return (
        pcp_indices,
        pcp_sub_locations,
        pcp_sub_correct,
        pcp_is_cdr,
        pcp_sample_family_dict,
    )


def get_site_subs_acc_df(
    sub_acc_tuple,
    numbering_dict=None,
):
    """
    Returns a dataframe that annotates for each site of observed substitution:
    the index of the PCP it belongs to,
    the site position in the amino acid sequence,
    whether the predicted amino acid substitution is correct.

    Parameters:
    subs_and_preds_tuple (tuple): 5-tuple of
                                  pcp_indices - list of indices to the reference PCP file,
                                  pcp_sub_locations - list of per-PCP lists of substitution locations,
                                  pcp_sub_correct - list of per-PCP lists whether amino acid prediction is correct,
                                  pcp_is_cdr = list of per-PCP lists whether amino acid site is in a CDR,
                                  pcp_sample_family_dict - dictionary mapping PCP index to (sample_id, family) 2-tuple.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): dataframe with columns 'pcp_index', 'site', 'correct', 'is_cdr'.
    """

    pcp_indices = sub_acc_tuple[0]
    pcp_sub_locations = sub_acc_tuple[1]
    pcp_sub_correct = sub_acc_tuple[2]
    pcp_is_cdr = sub_acc_tuple[3]
    pcp_sample_family_dict = sub_acc_tuple[4]

    pcp_index_col = []
    site_col = []
    pred_col = []
    is_cdr_col = []
    for i in range(len(pcp_indices)):
        nbkey = pcp_sample_family_dict[pcp_indices[i]]
        if (numbering_dict is not None) and (nbkey not in numbering_dict):
            continue

        for j in range(len(pcp_sub_locations[i])):
            site = pcp_sub_locations[i][j]
            pcp_index_col.append(pcp_indices[i])
            if numbering_dict is None:
                site_col.append(site)
            else:
                site_col.append(numbering_dict[nbkey][site])
            pred_col.append(pcp_sub_correct[i][j])
            is_cdr_col.append(pcp_is_cdr[i][j])

    output_df = pd.DataFrame(columns=["pcp_index", "site", "correct", "is_cdr"])
    output_df["pcp_index"] = pcp_index_col
    output_df["site"] = site_col
    output_df["correct"] = pred_col
    output_df["is_cdr"] = is_cdr_col

    return output_df


def plot_sites_subs_acc(
    df,
    counts_ax,
    subacc_ax,
    numbering_dict=None,
    fwr_color="#0072B2",
    cdr_color="#E69F00",
    logy=False,
):
    """
    Draws a figure of observed substitutions and the accuracy of amino acid predictions across PCPs in a dataset.
    The input dataframe requires three columns:
    'site' (site position -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'correct' (True/False whether the amino acid substitution prediction is correct)
    'is_cdr' (True/False whether the site is in a CDR)
    Each dataframe row corresponds to a site in a specific sequence.
    Assumes only sites with observed substitution are in the dataframe.

    Parameters:
    df (pd.DataFrame): dataframe of observed and predicted sites of substitution.
    counts_ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    subacc_ax (fig.ax): figure axis for plotting site accuracy. If None, plot is not drawn.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    fwr_color (str): color for FWR sites.
    cdr_color (str): color for CDR sites.
    logy (bool): whether to show y-axis in log-scale.

    Returns:
    A dictionary with results labeled:
    total_subacc (float): overall substitution accuracy of the dataset.
    site_subacc (list): per-site substitution accuracy of the dataset.
    """
    if numbering_dict is None:
        xvals = np.arange(np.min(df["site"]), np.max(df["site"]) + 1)
    else:
        xvals = numbering_dict[("reference", 0)]

    observed = []
    correct = []
    fwr_observed = []
    fwr_correct = []
    for site in xvals:
        site_df = df[df["site"] == site]
        nobs = site_df.shape[0]
        observed.append(nobs)
        ncorr = site_df[site_df["correct"]].shape[0]
        correct.append(ncorr)

        fwr_nobs = site_df[~site_df["is_cdr"]].shape[0]
        fwr_observed.append(fwr_nobs)
        fwr_ncorr = site_df[site_df["correct"] & (~site_df["is_cdr"])].shape[0]
        fwr_correct.append(fwr_ncorr)
    observed = np.array(observed)
    correct = np.array(correct)
    fwr_observed = np.array(fwr_observed)
    fwr_correct = np.array(fwr_correct)

    # compute substitution accuracy
    total_subacc = df[df["correct"]].shape[0] / df.shape[0]
    site_subacc = [c / o if o > 0 else -1 for c, o in zip(correct, observed)]

    if counts_ax is not None:
        counts_ax.bar(
            xvals,
            observed,
            width=1,
            facecolor="white",
            edgecolor=cdr_color,
            label="CDR",
        )

        counts_ax.bar(
            xvals,
            fwr_observed,
            width=1,
            facecolor="white",
            edgecolor=fwr_color,
            label="FWR",
        )

        counts_ax.bar(
            xvals,
            correct,
            width=1,
            color=cdr_color,
            edgecolor=cdr_color,
        )

        counts_ax.bar(
            xvals,
            fwr_correct,
            width=1,
            color=fwr_color,
            edgecolor=fwr_color,
        )

        if subacc_ax is None:
            if df.dtypes["site"] == "object":
                counts_ax.tick_params(axis="x", labelsize=7, labelrotation=90)
            else:
                counts_ax.tick_params(axis="x", labelsize=16)
            counts_ax.set_xlabel(
                "amino acid sequence position", fontsize=20, labelpad=10
            )
        counts_ax.tick_params(axis="y", labelsize=16)
        counts_ax.set_ylabel("number of substitutions", fontsize=20, labelpad=10)
        counts_ax.margins(x=0.01)

        if logy:
            counts_ax.set_yscale("log")

        counts_ax.legend(fontsize=15)

    if subacc_ax is not None:
        subacc_ax.plot(
            xvals,
            site_subacc,
            marker="d",
            markersize=4,
            linewidth=0,
            color="black",
        )

        if df.dtypes["site"] == "object":
            subacc_ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        else:
            subacc_ax.tick_params(axis="x", labelsize=16)
        subacc_ax.set_xlabel("amino acid sequence position", fontsize=20, labelpad=10)
        subacc_ax.tick_params(axis="y", labelsize=16)
        subacc_ax.set_ylabel("sub. acc.", fontsize=20, labelpad=10)
        subacc_ax.margins(x=0.01)
        subacc_ax.set_ylim(-0.02, 1.02)
        subacc_ax.grid(axis="y")

    return {
        "total_subacc": total_subacc,
        "site_subacc": site_subacc,
    }


def plot_sites_multi_subacc(
    df_list,
    markers,
    colors,
    modelnames,
    counts_ax,
    subacc_ax,
    numbering_dict=None,
    logy=False,
):
    """
    Draws a figure of per-site observed substitutions with number of correct predictions,
    and per-site substitution accuracies, for a list of models.
    The input dataframes requires two columns:
    'site' (site position of a substitution -- may be at the level of nucleotide, or codon, or amino acid, etc.)
    'correct' (True/False whether the amino acid substitution prediction is correct)
    Each dataframe row corresponds to a site in a specific sequence.
    Assumes only sites with observed substitution are in the dataframe.

    Parameters:
    df_list (list of pd.DataFrame): list of dataframes of observed and predicted sites of substitution.
    markers (list): list of marker styles for each model
    colors (list): list of colors for each model
    modelnames (list): list of model names for each model
    counts_ax (fig.ax): figure axis for plotting site counts. If None, plot is not drawn.
    subacc_ax (fig.ax): figure axis for plotting site accuracy. If None, plot is not drawn.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.
    logy (bool): whether to show y-axis in log-scale.

    """
    if numbering_dict is None:
        xvals = np.arange(np.min(df_list[0]["site"]), np.max(df_list[0]["site"]) + 1)
    else:
        xvals = numbering_dict[("reference", 0)]

    observed = []
    correct_list = []
    for i in range(len(df_list)):
        df = df_list[i]
        correct = []
        for site in xvals:
            site_df = df[df["site"] == site]
            if i == 0:
                nobs = site_df.shape[0]
                observed.append(nobs)
            ncorr = site_df[site_df["correct"]].shape[0]
            correct.append(ncorr)
        correct = np.array(correct)
        correct_list.append(correct)
    observed = np.array(observed)

    # compute substitution accuracy
    site_subacc_list = [
        [c / o if o > 0 else -1 for c, o in zip(correct, observed)]
        for correct in correct_list
    ]

    if counts_ax is not None:
        counts_ax.plot(
            xvals,
            observed,
            marker="o",
            markersize=6,
            linewidth=2,
            color="black",
            label="observed",
        )

        for correct, marker, color, modelname in zip(
            correct_list, markers, colors, modelnames
        ):
            counts_ax.plot(
                xvals,
                correct,
                marker=marker,
                markersize=6,
                fillstyle="none",
                linewidth=2,
                color=color,
                label=modelname,
            )

        if subacc_ax is None:
            if df.dtypes["site"] == "object":
                counts_ax.tick_params(axis="x", labelsize=7, labelrotation=90)
            else:
                counts_ax.tick_params(axis="x", labelsize=16)
            counts_ax.set_xlabel(
                "amino acid sequence position", fontsize=20, labelpad=10
            )
        counts_ax.tick_params(axis="y", labelsize=16)
        counts_ax.set_ylabel("number of substitutions", fontsize=20, labelpad=10)
        counts_ax.margins(x=0.01)

        if logy:
            counts_ax.set_yscale("log")

        counts_ax.legend(fontsize=15)

    if subacc_ax is not None:
        for site_subacc, marker, color in zip(site_subacc_list, markers, colors):
            subacc_ax.plot(
                xvals,
                site_subacc,
                marker=marker,
                markersize=6,
                fillstyle="none",
                linewidth=2,
                color=color,
            )

        if df.dtypes["site"] == "object":
            subacc_ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        else:
            subacc_ax.tick_params(axis="x", labelsize=16)
        subacc_ax.set_xlabel("amino acid sequence position", fontsize=20, labelpad=10)
        subacc_ax.tick_params(axis="y", labelsize=16)
        subacc_ax.set_ylabel("sub. acc.", fontsize=20, labelpad=10)
        subacc_ax.margins(x=0.01)
        subacc_ax.set_ylim(-0.02, 1.02)
        subacc_ax.grid(axis="y")


def annotate_site_csp_df(
    df,
    pcp_df,
    numbering_dict=None,
):
    """Create a DataFrame with modified or additional annotations for site numbering and
    CDR label. The input DataFrame describes a site and conditional substitution
    probability (CSP) in each row for an amino acid. The `site` corresponds to a site of
    substitution, and there are 20 rows for each site to cover all amino acids. Each row
    is expected to have the `pcp_index` column, indicating the index of the PCP the site
    belongs to. `prob` denotes a CSP and `aa` denotes the index in AA_STR_SORTED of the
    corresponding amino acid. `is_target` denotes if the amino acid is the observed
    substitution target.

    Parameters:
    df (pd.DataFrame): site CSPs DataFrame.
    pcp_df (pd.DataFrame): PCP file of the dataset.
    numbering_dict (dict): mapping (sample_id, family) to numbering list.

    Returns:
    output_df (pd.DataFrame): a new dataframe with columns pcp_index, site, prob, aa, mutation, is_cdr.
    Notes:
    The `site` column will be changed if a numbering scheme was specified by numbering_dict.
    `mutation` column is the same content as `is_target`, the column name change allows it to be usable
    by various plotting functions. `aa` is in terms of alphabet symbols.
    The size of output_df may be different from input df if there were PCPs that did not have a valid ANARCI numbering.
    """

    site_col = []
    is_cdr_col = []

    pcp_groups = df.groupby("pcp_index")
    for pcp_index in df["pcp_index"].drop_duplicates():
        pcp_row = pcp_df.loc[pcp_index]

        pcp_group_df = pcp_groups.get_group(pcp_index)

        pcp_is_cdr = pcp_sites_cdr_annotation(pcp_row)

        for site in pcp_group_df["site"].drop_duplicates():
            if site >= len(pcp_is_cdr):
                print(pcp_index, site, len(pcp_is_cdr))
                continue
            if numbering_dict is None:
                site_col.append([site] * 20)
            else:
                nbkey = tuple(pcp_row[["sample_id", "family"]])
                if nbkey in numbering_dict:
                    site_col.append([numbering_dict[nbkey][site]] * 20)
                else:
                    # Assign sites as "None", marking them for exclusion from output.
                    site_col.append(["None"] * 20)

            is_cdr_col.append([pcp_is_cdr[site]] * 20)

    output_df = pd.DataFrame(
        columns=["pcp_index", "site", "prob", "aa", "mutation", "is_cdr"]
    )
    output_df["pcp_index"] = df["pcp_index"].to_numpy()
    output_df["site"] = np.concatenate(site_col)
    output_df["prob"] = df["prob"].to_numpy()
    output_df["aa"] = [AA_STR_SORTED[i] for i in df["aa"]]
    output_df["mutation"] = df["is_target"].to_numpy()
    output_df["is_cdr"] = np.concatenate(is_cdr_col)

    if numbering_dict is None:
        return output_df
    else:
        return output_df[output_df["site"] != "None"]


def is_imgt_cdr(site):
    """Determines whether an amino acid site is in a CDR according to IMGT numbering.

    Parameters:
    site (str): IMGT number of an amino acid site.

    Returns:
    True or False whether the site is in a CDR.
    """
    IMGT_CDR1 = (27, 38)
    IMGT_CDR2 = (56, 65)
    IMGT_CDR3 = (105, 117)

    # Note: IMGT uses decimals for insertions (e.g. '111.3')
    if "." in site:
        sitei = int(site.split(".")[0])
    else:
        sitei = int(site)

    return (
        (sitei >= IMGT_CDR1[0] and sitei <= IMGT_CDR1[1])
        or (sitei >= IMGT_CDR2[0] and sitei <= IMGT_CDR2[1])
        or (sitei >= IMGT_CDR3[0] and sitei <= IMGT_CDR3[1])
    )


def get_numbering_dict(anarci_path, pcp_df=None, verbose=False, checks="imgt"):
    """Process ANARCI output to make site numbering lists for each clonal family.

    Parameters:
    anarci_path (str): path to ANARCI output for sequence numbering.
    pcp_df (pd.Dataframe): PCP file to filter for relevant clonal families and check ANARCI sequence lengths.
    verbose (bool): whether to print (sample ID, family ID) info when clonal family is excluded.
    checks (str): perform checks and updates for a specified numbering scheme.
                  Currently, 'imgt' is the only input that has an effect.

    Returns:
    Two dictionaries where the keys are 2-tuples of (sample_id, family).
    The first dictionary have values that are lists of numberings for each site in the clonal family.
    Note that the numberings are lists of strings.
    The dictionary also has an entry with key ('reference', 0) and value the list of all site numberings,
    to be used for setting x-axis tick labels when plotting.
    The second dictionary consists of clonal families that excluded from the first due to issues with the ANARCI output.
    The values describe the reasons for exclusion.
    """
    if anarci_path is None:
        return None

    numbering_dict = {}
    exclusion_dict = {}

    anarci_df = pd.read_csv(anarci_path)

    # assumes numbering starts at column 13 in ANARCI output
    numbering_cols = list(anarci_df.columns[13:])
    numbering_used = [False] * len(numbering_cols)

    for i, row in anarci_df.iterrows():
        # assumes clonal family ID has format "{sample_id}|{family}|{seq_name}"
        cfinfos = row["Id"].split("|")
        sample_id = cfinfos[0]
        family = cfinfos[1]

        seqlist = [row[col] for col in numbering_cols]
        numbering = [nn for nn, aa in zip(numbering_cols, seqlist) if aa != "-"]

        if checks == "imgt":
            # For IMGT, numbered insertions can only be 111.* or 112.*.
            # Other numbered insertions come from ANARCI and the clonal family will be excluded
            exclude = False
            for nn in numbering:
                if "." in nn and nn[:3] != "111" and nn[:3] != "112":
                    exclusion_dict[(sample_id, family)] = (
                        f"Invalid IMGT insertion: {nn}"
                    )
                    if verbose:
                        print(f"Invalid IMGT insertion: {nn}", sample_id, family)
                    exclude = True
                    break
            if exclude:
                continue

        if pcp_df is not None:
            # Check if clonal family is in PCP file, and that ANARCI preserved sequence length.
            # If not, exclude clonal family from output.
            test_df = pcp_df[
                (pcp_df["sample_id"] == sample_id) & (pcp_df["family"] == family)
            ]
            if test_df.shape[0] == 0:
                continue
            else:
                pcp_row = test_df.iloc[0]
                test_seq = translate_sequence(pcp_row["parent"])
                if len(test_seq) != len(numbering):
                    exclusion_dict[(sample_id, family)] = "ANARCI seq length mismatch!"
                    if verbose:
                        print("ANARCI seq length mismatch!", sample_id, family)
                    continue

                if checks == "imgt":
                    # Check CDR annotation in PCP file is consistent with IMGT numbering.
                    # If not, exclude the clonal family.
                    cdr_anno = pcp_sites_cdr_annotation(pcp_row)

                    exclude = False
                    for nn, is_cdr in zip(numbering, cdr_anno):
                        if is_imgt_cdr(nn) != is_cdr:
                            exclusion_dict[(sample_id, family)] = (
                                "IMGT mismatch with CDR annotation!"
                            )
                            if verbose:
                                print(
                                    "IMGT mismatch with CDR annotation!",
                                    sample_id,
                                    family,
                                )
                            exclude = True
                            break
                    if exclude:
                        continue

        numbering_dict[(sample_id, family)] = numbering

        # keep track of which site numbers are used
        for nn in numbering:
            numbering_used[numbering_cols.index(nn)] = True

    # make a numbering reference of site numbers that are used
    numbering_dict[("reference", 0)] = [
        nn for nn, used in zip(numbering_cols, numbering_used) if used
    ]

    return (numbering_dict, exclusion_dict)


def pcp_sites_cdr_annotation(pcp_row):
    """Annotations for CDR or not for all sites in a PCP.

    Parameters:
    pcp_row (pd.Series): A row from the corresponding PCP file

    Returns:
    An (ordered) list of booleans for whether each site is in the CDR or not. The list is the same length as the sequence.
    """
    cdr1 = (
        pcp_row["cdr1_codon_start"] // 3,
        pcp_row["cdr1_codon_end"] // 3,
    )
    cdr2 = (
        pcp_row["cdr2_codon_start"] // 3,
        pcp_row["cdr2_codon_end"] // 3,
    )
    cdr3 = (
        pcp_row["cdr3_codon_start"] // 3,
        pcp_row["cdr3_codon_end"] // 3,
    )

    return [
        (
            True
            if (i >= cdr1[0] and i <= cdr1[1])
            or (i >= cdr2[0] and i <= cdr2[1])
            or (i >= cdr3[0] and i <= cdr3[1])
            else False
        )
        for i in range(len(pcp_row["parent"]) // 3)
    ]


def pcp_sites_regions(pcp_row):
    """Annotations for CDR/FWR for all sites in a PCP.

    Parameters:
    pcp_row (pd.Series): A row from the corresponding PCP file

    Returns:
    An (ordered) list of strings denoting the region for each site. The list is the same length as the sequence.
    """
    cdr1 = (
        pcp_row["cdr1_codon_start"] // 3,
        pcp_row["cdr1_codon_end"] // 3,
    )
    cdr2 = (
        pcp_row["cdr2_codon_start"] // 3,
        pcp_row["cdr2_codon_end"] // 3,
    )
    cdr3 = (
        pcp_row["cdr3_codon_start"] // 3,
        pcp_row["cdr3_codon_end"] // 3,
    )

    regions = []
    for i in range(len(pcp_row["parent"]) // 3):
        if i < cdr1[0]:
            regions.append("FWR1")
        elif i <= cdr1[1]:
            regions.append("CDR1")
        elif i < cdr2[0]:
            regions.append("FWR2")
        elif i <= cdr2[1]:
            regions.append("CDR2")
        elif i < cdr3[0]:
            regions.append("FWR3")
        elif i <= cdr3[1]:
            regions.append("CDR3")
        else:
            regions.append("FWR4")

    return regions
