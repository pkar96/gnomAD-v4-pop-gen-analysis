import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from scipy.integrate import trapezoid, cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sc
from scipy.special import logsumexp
import glob
import os
import math
import joblib
from joblib import Parallel, delayed, parallel_backend
import random


"""
missense_pipeline_single.py

Single-file pipeline to:
1) Load a missense dataframe (.pkl).
2) Filter variants (allele_count <= max, score non-NA).
3) Build MR_rank (rank of MR bins/values).
4) Fit per-gene prior parameters (c, beta, sigma) by MLE using SFS (missense SFS).
5) Compute per-variant medians:
   - s_prior_{score_col}: prior median of s (positive; i.e. -s in your previous notation)
   - s_{score_col}: posterior median of s (positive)
6) Optionally resolve duplicates with a LoF reference table.
7) Save final dataframe to output .pkl.

Requirements: numpy, pandas, scipy, joblib
"""


# ----------------------------
# I/O helpers
# ----------------------------
def load_df_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {pkl_path}, got {type(obj)}")
    return obj


def save_df_pkl(df, pkl_path):
    os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)


def load_sfs(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ----------------------------
# Data prep
# ----------------------------
def filter_variants(df, score_col, allele_count_max=5000):
    if score_col not in df.columns:
        raise ValueError(f"score_col '{score_col}' not found in df")
    needed = ["gene_id", "allele_count", "MR", score_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[(df["allele_count"].notna()) & (df["allele_count"] <= allele_count_max) & (df[score_col].notna())].copy()
    return out


def inverse_normal_transform(series):
    # Rank data
    ranks = series.rank(method='average')
    # Convert to uniform quantiles in (0, 1)
    quantiles = (ranks - 0.5) / len(series)
    # Map to normal distribution
    return norm.ppf(quantiles)


def add_mr_rank(df, mr_col="MR", out_col="MR_rank"):
    unique_mr = df[mr_col].dropna().unique()
    mr_rank_dict = {value: rank for rank, value in enumerate(sorted(unique_mr))}
    df[out_col] = df[mr_col].map(mr_rank_dict).astype("int32")
    return df


def construct_gene_dfs(df):
    # dict: gene_id -> df (reset index)
    return {gid: g.reset_index(drop=True) for gid, g in df.groupby("gene_id", sort=False)}


# ----------------------------
# Likelihood + MLE per gene
# ----------------------------
def gene_log_likelihood_trap_no_folding(df_gene, s_values_neg, sfs, c, beta, sigma, score_col):
    """
    df_gene must have columns: score_col, allele_count, MR_rank
    s_values_neg: negative s grid, shape (S,)
    sfs: array shape (n_mu, S+1, n_AC) or (n_mu, S+something, n_AC)
         We will use sfs[mu_idx, :-1, ac_idx] as in your code.
    """
    # Convert to positive s for lognormal on s = -s_values_neg
    s_pos = -np.asarray(s_values_neg, dtype=float)  # (S,)
    log_s = np.log10(s_pos)
    zmin, zmax = log_s[0], log_s[-1]
    ln10 = np.log(10)

    sc_val = df_gene[score_col].to_numpy()
    mu_idx = df_gene["MR_rank"].astype(int).to_numpy()
    ac_idx = df_gene["allele_count"].astype(int).to_numpy()

    # prior mean in z = log10(s) via sigmoid(score)
    mu_z = zmin + (zmax - zmin) * (1.0 / (1.0 + np.exp(-beta * (sc_val - c))))
    mu_z = np.clip(mu_z, -300, 300)

    shape = sigma * ln10
    scale = 10 ** mu_z

    shape = np.atleast_1d(shape)

    # continuous pdf in s-space
    pdf_s = lognorm.pdf(s_pos[None, :], s=shape[:, None], scale=scale[:, None])  # (N,S)

    # delta masses outside [zmin,zmax]
    dl = norm.cdf(zmin, loc=mu_z, scale=sigma)
    du = norm.sf(zmax, loc=mu_z, scale=sigma)

    cont_mass = trapezoid(pdf_s, s_pos, axis=1)
    tot_mass = dl + cont_mass + du

    pdf_cont = pdf_s / tot_mass[:, None]
    dl_n = dl / tot_mass
    du_n = du / tot_mass

    # likelihood curves
    sfs_vals = sfs[mu_idx, :-1, ac_idx]  # (N,S)

    cont_like = trapezoid(pdf_cont * sfs_vals, s_pos, axis=1)
    low_like = dl_n * sfs_vals[:, 0]
    high_like = du_n * sfs_vals[:, -1]

    marg_like = cont_like + low_like + high_like
    total_ll = float(np.sum(np.log(marg_like + 1e-300)))
    return total_ll


def optimize_gene_mle(gene_id, df_gene, s_values_neg, sfs_mle, score_col,
                      init_c=4.0, init_beta=1.0, init_log_sigma=np.log(0.5551),
                      bounds=None, maxiter=400, fatol=1e-8,
                      out_dir=None):
    """
    Returns dict with c,beta,sigma,max_log_likelihood,success,n_sites.
    Optionally writes a per-gene .txt line in out_dir.
    """
    if bounds is None:
        # derive dx from mle grid for log_sigma lower bound
        log_s_grid = np.log10(-np.asarray(s_values_neg))
        dx = float(log_s_grid[1] - log_s_grid[0])
        bounds = [(0, 8), (0, 10), (np.log(dx), np.log(10.0))]  # (c, beta, log_sigma)

    x0 = np.array([init_c, init_beta, init_log_sigma], dtype=float)

    def neg_ll(params):
        c, beta, log_sigma = params
        sigma = float(np.exp(log_sigma))
        return -gene_log_likelihood_trap_no_folding(df_gene, s_values_neg, sfs_mle,
                                                    float(c), float(beta), sigma, score_col)

    res = minimize(
        neg_ll,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": int(maxiter), "fatol": float(fatol)}
    )

    c, beta, log_sigma = res.x
    sigma = float(np.exp(log_sigma))
    out = {
        "gene_id": gene_id,
        "c": float(c),
        "beta": float(beta),
        "sigma": float(sigma),
        "max_log_likelihood": float(-res.fun),
        "success": bool(res.success),
        "n_sites": int(len(df_gene)),
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{gene_id}.txt"), "w") as f:
            f.write(
                f"{gene_id}\t{out['c']:.6g}\t{out['beta']:.6g}\t{out['sigma']:.6g}\t{out['max_log_likelihood']:.6g}\t{out['success']}\t{out['n_sites']}\n"
            )

    return out


def fit_all_genes_mle(gene_dfs, s_values_neg, sfs_mle, score_col,
                      init_c=4.0, init_beta=1.0, init_log_sigma=np.log(0.5551),
                      bounds=None, maxiter=400, fatol=1e-8,
                      n_jobs=-1, backend="multiprocessing", verbose=10,
                      out_dir=None):
    gene_lengths = {gid: len(g) for gid, g in gene_dfs.items()}
    sorted_gene_ids = sorted(gene_lengths, key=gene_lengths.get)

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(optimize_gene_mle)(
            gid, gene_dfs[gid], s_values_neg, sfs_mle, score_col,
            init_c, init_beta, init_log_sigma,
            bounds, maxiter, fatol, out_dir
        )
        for gid in sorted_gene_ids
    )

    return pd.DataFrame(results)


# ----------------------------
# Variant-level medians (prior/posterior)
# ----------------------------
def variant_medians_for_gene(scores, allele_count, mr_rank, sfs_variant, c, beta, sigma, s_values_neg):
    """
    Returns (s_median_prior, s_median_post) for each variant.
    s returned is positive (same as your s_grid = 10**log_s), i.e. -s in old sign convention.
    """
    s_grid = -np.asarray(s_values_neg, dtype=float)  # (S,)
    log_s = np.log10(s_grid)
    zmin, zmax = log_s[0], log_s[-1]
    ln10 = np.log(10)
    
    scores = np.asarray(scores, dtype=float)
    allele_count = np.asarray(allele_count, dtype=int)
    mr_rank = np.asarray(mr_rank, dtype=int)

    # prior params in z-space
    mu_z = zmin + (zmax - zmin) * (1.0 / (1.0 + np.exp(-beta * (scores - c))))
    mu_z = np.clip(mu_z, -300, 300)
    shape = sigma * ln10
    scale = 10 ** mu_z

    shape = np.atleast_1d(shape)

    pdf_s = lognorm.pdf(s_grid[None, :], s=shape[:, None], scale=scale[:, None])  # (N,S)

    dl = norm.cdf(zmin, loc=mu_z, scale=sigma)
    du = norm.sf(zmax, loc=mu_z, scale=sigma)

    cont_mass = trapezoid(pdf_s, s_grid, axis=1)
    tot_mass = dl + cont_mass + du

    pdf_cont_prior = pdf_s / tot_mass[:, None]
    dl_n_prior = dl / tot_mass
    du_n_prior = du / tot_mass

    # likelihood
    sfs_vals = sfs_variant[mr_rank, :-1, allele_count]  # (N,S)

    # posterior unnormalized
    unnorm = pdf_cont_prior * sfs_vals
    post_lower = dl_n_prior * sfs_vals[:, 0]
    post_upper = du_n_prior * sfs_vals[:, -1]

    cont_mass_post = trapezoid(unnorm, s_grid, axis=1)
    tot_mass_post = post_lower + cont_mass_post + post_upper

    pdf_cont_post = unnorm / tot_mass_post[:, None]
    dl_n_post = post_lower / tot_mass_post
    du_n_post = post_upper / tot_mass_post

    # CDFs
    cdf_prior_cont = cumulative_trapezoid(pdf_cont_prior, s_grid, axis=1, initial=0.0)
    cdf_prior = dl_n_prior[:, None] + cdf_prior_cont
    cdf_prior[:, -1] += du_n_prior

    cdf_post_cont = cumulative_trapezoid(pdf_cont_post, s_grid, axis=1, initial=0.0)
    cdf_post = dl_n_post[:, None] + cdf_post_cont
    cdf_post[:, -1] += du_n_post

    idx_med_prior = (cdf_prior >= 0.5).argmax(axis=1)
    idx_med_post = (cdf_post >= 0.5).argmax(axis=1)

    return s_grid[idx_med_prior], s_grid[idx_med_post]



def _one_gene_attach_medians(gid, df_gene, summary_idx, score_col, sfs_variant, s_values_neg, out_prior_col, out_post_col):
    if gid not in summary_idx.index:
        return None

    row = summary_idx.loc[gid]
    c = float(row["c"])
    beta = float(row["beta"])
    sigma = float(row["sigma"])

    sc = df_gene[score_col].to_numpy()
    ac = df_gene["allele_count"].astype(int).to_numpy()
    mr = df_gene["MR_rank"].astype(int).to_numpy()

    s_med_prior, s_med_post = variant_medians_for_gene(sc, ac, mr, sfs_variant, c, beta, sigma, s_values_neg)

    out = df_gene.copy()
    out[out_prior_col] = s_med_prior.astype("float32")
    out[out_post_col]  = s_med_post.astype("float32")
    return out

def attach_medians_all_genes(gene_dfs, df_summary, score_col, sfs_variant, s_values_neg,
                             out_prior_col, out_post_col,
                             n_jobs=-1, backend="multiprocessing", verbose=10):

    summary_idx = df_summary.set_index("gene_id")

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_one_gene_attach_medians)(
            gid, gdf, summary_idx, score_col, sfs_variant, s_values_neg, out_prior_col, out_post_col
        )
        for gid, gdf in gene_dfs.items()
    )

    results = [r for r in results if r is not None and len(r) > 0]
    return pd.concat(results, ignore_index=True)


# ----------------------------
# Optional duplicate resolution
# ----------------------------
def resolve_duplicates(df_main, df_ref, key_cols, score_main, score_ref):
    """
    Keep one row per key_cols group:
      - if all score_ref NaN in group: keep row with max score_main
      - else keep row with max score_ref
    """
    df_merged = df_main.merge(df_ref[["gene_id", score_ref]], on="gene_id", how="left")

    case_counts = {"all_nan": 0, "some_valid": 0, "all_valid": 0}

    def pick_best(g):
        n_total = len(g)
        n_nan = int(g[score_ref].isna().sum())
        if n_nan == n_total:
            case_counts["all_nan"] += 1
            return g.loc[g[score_main].idxmax()]
        elif n_nan == 0:
            case_counts["all_valid"] += 1
            return g.loc[g[score_ref].idxmax()]
        else:
            case_counts["some_valid"] += 1
            return g.loc[g[score_ref].idxmax()]

    dup_mask = df_merged.duplicated(subset=key_cols, keep=False)
    df_dups = df_merged[dup_mask]
    df_unique = df_merged[~dup_mask]

    df_resolved = df_dups.groupby(key_cols, group_keys=False).apply(pick_best)
    df_final = pd.concat([df_unique, df_resolved], ignore_index=True)

    # drop the merged ref score column
    df_final = df_final.drop(columns=[score_ref], errors="ignore")
    return df_final, case_counts


# ----------------------------
# Main runner
# ----------------------------
def run_pipeline(input_file, output_file, score_col, allele_count_max=5000,
    sfs_mle_pkl="/n/data2/hms/dbmi/sunyaev/lab/pkar/demography_SFS/SFS_all_missense.pkl",
    sfs_variant_pkl="/n/data2/hms/dbmi/sunyaev/lab/pkar/demography_SFS/SFS_all.pkl",
    # MLE params
    init_c=4.0,
    init_beta=1.0,
    init_log_sigma=np.log(0.5551),
    bounds=None,
    maxiter=400,
    fatol=1e-8,
    # parallel
    n_jobs=-1,
    backend="multiprocessing",
    verbose=10,
    # outputs
    gene_mle_out_dir=None,
    # optional LoF dedup
    lof_ref_path=None,
    lof_ref_score_col="Posterior_CI10_lower",
    key_cols=None
):
    t0 = time.time()

    if key_cols is None:
        key_cols = ["#CHROM", "POS", "REF", "ALT"]

    # 1) load
    df_orig = pd.read_csv(input_file, sep="\t", compression="gzip")

    # 2) filter + MR rank
    df = filter_variants(df_orig, score_col, allele_count_max=allele_count_max)
    df = add_mr_rank(df, "MR", "MR_rank")

    if score_col == "AlphaMissense":
        df[f"{score_col}_norm"] = df.groupby("gene_id")[score_col].transform(inverse_normal_transform)
        score_col = f"{score_col}_norm"

    # 3) per-gene
    gene_dfs = construct_gene_dfs(df)

    # 4) load SFSs
    sfs_mle = load_sfs(sfs_mle_pkl)
    s_values_mle = -np.logspace(-6, 2, num=100)
    sfs_variant = load_sfs(sfs_variant_pkl)
    s_values_variant = -np.logspace(-6, 2, num=500)

    # 5) MLE per gene
    if gene_mle_out_dir is None:
        gene_mle_out_dir = os.path.join(os.path.dirname(output_file) or ".", f"gene_mle_results_{score_col}")

    df_summary = fit_all_genes_mle(
        gene_dfs, s_values_mle, sfs_mle, score_col,
        init_c=init_c, init_beta=init_beta, init_log_sigma=init_log_sigma,
        bounds=bounds, maxiter=maxiter, fatol=fatol,
        n_jobs=n_jobs, backend=backend, verbose=verbose,
        out_dir=gene_mle_out_dir
    )

    # 6) attach medians
    out_prior_col = f"s_prior_{score_col}"
    out_post_col = f"s_{score_col}"

    df_all = attach_medians_all_genes(
        gene_dfs, df_summary, score_col, sfs_variant, s_values_variant,
        out_prior_col, out_post_col,
        n_jobs=n_jobs, backend=backend, verbose=verbose
    )

    # 7) optional dedup with LoF ref
    case_counts = None
    if lof_ref_path is not None:
        df_lof = pd.read_csv(lof_ref_path, sep="\t", compression="gzip")
        df_all, case_counts = resolve_duplicates(
            df_main=df_all,
            df_ref=df_lof,
            key_cols=key_cols,
            score_main=out_prior_col,        # same as your previous default
            score_ref=lof_ref_score_col,
        )
        print("Duplicate resolution summary:", case_counts)

    # 8) save
    save_df_pkl(df_all, output_file)

    t1 = time.time()
    print(f"Saved: {output_file}")
    print(f"Columns added: {out_prior_col}, {out_post_col}")
    print(f"Time: {t1 - t0:.2f} s")
    return df_all, df_summary, case_counts


# # ----------------------------
# # CLI (optional)
# # ----------------------------
# if __name__ == "__main__":
#     import argparse

#     p = argparse.ArgumentParser()
#     p.add_argument("--input-pkl", required=True)
#     p.add_argument("--output-pkl", required=True)
#     p.add_argument("--score-col", required=True)
#     p.add_argument("--allele-count-max", type=int, default=5000)
#     p.add_argument("--n-jobs", type=int, default=-1)
#     p.add_argument("--backend", default="multiprocessing")
#     p.add_argument("--verbose", type=int, default=10)

#     p.add_argument("--sfs-mle-pkl", default="/n/data2/hms/dbmi/sunyaev/lab/pkar/demography_SFS/SFS_all_missense.pkl")
#     p.add_argument("--sfs-variant-pkl", default="/n/data2/hms/dbmi/sunyaev/lab/pkar/demography_SFS/SFS_all.pkl")

#     p.add_argument("--gene-mle-out-dir", default=None)

#     p.add_argument("--lof-ref-path", default=None)
#     p.add_argument("--lof-ref-score-col", default="Posterior_CI10_lower")

#     args = p.parse_args()

#     run_pipeline(
#         input_pkl=args.input_pkl,
#         output_pkl=args.output_pkl,
#         score_col=args.score_col,
#         allele_count_max=args.allele_count_max,
#         sfs_mle_pkl=args.sfs_mle_pkl,
#         sfs_variant_pkl=args.sfs_variant_pkl,
#         n_jobs=args.n_jobs,
#         backend=args.backend,
#         verbose=args.verbose,
#         gene_mle_out_dir=args.gene_mle_out_dir,
#         lof_ref_path=args.lof_ref_path,
#         lof_ref_score_col=args.lof_ref_score_col,
#     )
