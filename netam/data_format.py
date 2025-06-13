"""Constants defining data format for pcp dataframes and other data."""

from collections import defaultdict


_pcp_df_differentiated_columns = {
    "parent": str,
    "child": str,
    "v_gene": str,
    # These should be nullable, because they may be missing in combined
    # heavy/light bulk dataframes.
    "cdr1_codon_start": "Int64",
    "cdr1_codon_end": "Int64",
    "cdr2_codon_start": "Int64",
    "cdr2_codon_end": "Int64",
    "cdr3_codon_start": "Int64",
    "cdr3_codon_end": "Int64",
    "j_gene": str,
    "v_family": str,
}

_pcp_df_undifferentiated_columns = {
    "sample_id": str,
    "family": str,
    "parent_name": str,
    "child_name": str,
    "branch_length": float,
    "depth": int,
    "distance": float,
    "parent_is_naive": bool,
    "child_is_leaf": bool,
}

_all_pcp_df_columns = (
    defaultdict(lambda: "object")
    | {col + "_heavy": dtype for col, dtype in _pcp_df_differentiated_columns.items()}
    | {col + "_light": dtype for col, dtype in _pcp_df_differentiated_columns.items()}
    | _pcp_df_undifferentiated_columns
)

_required_pcp_df_columns = ("parent", "child", "v_gene")
