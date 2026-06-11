# --- make parent folder importable ---
import sys
# --- usual imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.integrate import quad
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.stats import gmean, linregress, norm, beta, uniform, lognorm
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.integrate import simpson
from scipy import stats
import glob
import os
import math
import re
import csv
from tqdm import tqdm
import time
import joblib
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import pickle
import random
import json


def choose_ac_an_cols(demography):
    """
    Get the column name of allele count (ac_col) and allele number (an_col)
    """
    if demography == "all":
        ac_col = "allele_count"
        AN_col = "allele_number"
    elif demography == "schraiber_et_al":
        ac_col = "AC_NFE"
        AN_col = "AN_NFE"
    else:
        ac_col = f"AC_{demography}"
        AN_col = f"AN_{demography}"

    return ac_col, AN_col



def construct_per_gene_dfs(df):
    """
    Constructs a dictionary of dataframes for each gene with the relevant columns for analysis.
    """
    gene_dfs = {gene_id: group.reset_index(drop=True) for gene_id, group in df.groupby("gene_id")}
    return gene_dfs



def gene_specific_LL_full(curr_gene_id, gene_df, Bell_tensor, pnull_dict, AC_col):
    '''
    Returns the whole likelihood function as a function of s
    curr_gene_id: Ensembl id of the current gene
    gene_df : Dataframe storing info for curr_gene_id gene
    Bell_tensor: Same as SFSs, stores the predicted SFS for all Roulette values and 
                 500 selection values from (-10**-6 to -100) + neutral (s = 0) for allele counts from 0 to 5000
    pnull_dict: Dictionary with three items "stop_gained", "splice_donor_variant" and "splice_acceptor_variant". Values stores the prob of misannotation
    AC_col: Stores the allele count column name for the demography being analyzed
    '''
    # Initialize the LL vector for the SFS
    LL_vec = np.zeros(Bell_tensor.shape[1] - 1)  # Initialize the LL vector for the SFS
    
    for annotation, pnull in pnull_dict.items():
        annotation_df = gene_df[gene_df["most_severe_consequence2"] == annotation].reset_index(drop=True)
        mu_inds = annotation_df["MR_rank"].astype(int).values
        ACs = annotation_df[AC_col].astype(int).values
        
        p_neuts = Bell_tensor[mu_inds, -1, ACs] # Neutral probabilities
        p_Ss = Bell_tensor[mu_inds, :-1, ACs] # Selection probabilities
        
        # Calculate the log-likelihood for each variant
        Ls = np.log10(pnull * p_neuts[:, None] + (1 - pnull) * p_Ss).T
        # Sum the log-likelihoods across all variants for each annotation
        LLs = np.sum(Ls, axis=1)
        LL_vec += LLs
    return curr_gene_id, LL_vec



def optimize_gene_s_ret_logL(curr_gene_id, gene_df, Bell_tensor, pnull_dict, AC_col):
    """
    Compute the full LL curve and return the maximum LL for this gene.
    """
    _, LL_vec = gene_specific_LL_full(curr_gene_id, gene_df, Bell_tensor, pnull_dict, AC_col)
    return np.max(LL_vec)



# Likelihood for pnull estimate combining over 
def p_misannot_s(params, gene_dfs, Bell_tensor, AC_col):

    stop_gain, splice_donor, splice_acceptor = params
    pnull_dict = {"stop_gained": stop_gain, "splice_donor_variant": splice_donor, "splice_acceptor_variant": splice_acceptor}

    results = Parallel(n_jobs=-1)(
        delayed(optimize_gene_s_ret_logL)(
            curr_gene_id,
            gene_dfs[curr_gene_id],
            Bell_tensor,
            pnull_dict,
            AC_col
        )
        for curr_gene_id in gene_dfs.keys()
    )

    neg_loglikelihood = -np.sum(results)

    return neg_loglikelihood

    

def likelihood_LoF(LoF_filename, demography, SFS_file, outdir_pnull, pkl_LL_folder, kmax = 5000):
    """
    Main function that calculates the likelihood(selection s) of observing SFS of LoFs
    Input:
        LoF_filename  :File containing a table of allele counts, allele number, mutation rate MR_rank, Ensembl gene_id, annotation (stop-gained, splice donor or splice acceptor) of LoFs. 
                       The allele counts and allele numbers are in columns, AC_{demography} and AN_{demography} respectively. 
                       MR_rank contains the row numbers of SFS hash table that we use in likelihood function.
                       annotation is stored in "most_severe_consequence2" column. 
        demography    : Demography for which we are calculating the likelihood
        SFS_file      : contains SFS hashtable. (len(mut rate bins), len(selection grid), (ac = 0-kmax)) shape. See Make_SFS_demography.ipynb for more details.
    Output:
        outdir_pnull  : Stores a csv file in outdir_pnull folder storing probability of misannotation values (pnull) for a demography.
        pkl_LL_folder : Stores the log10(Likelihood) as function of s for all genes and given demography as a pickle file.
    """

    # Get the LoF file loaded and get the columns that store allele counts
    df_LoF_orig = pd.read_csv(LoF_filename, sep="\t", compression="gzip")
    ac_col, AN_col = choose_ac_an_cols(demography)

    # Filters out the CES genes and variants with allele counts >5000
    CE_genes = np.array(['ENSG00000078369', 'ENSG00000157933', 'ENSG00000198793', 'ENSG00000117713', 'ENSG00000117400', 'ENSG00000213281', 'ENSG00000143379', 'ENSG00000143622',
                         'ENSG00000135829', 'ENSG00000117139', 'ENSG00000171862', 'ENSG00000108055', 'ENSG00000108061', 'ENSG00000148737', 'ENSG00000066468', 'ENSG00000174775',
                         'ENSG00000184937', 'ENSG00000149480', 'ENSG00000168066', 'ENSG00000175115', 'ENSG00000160584', 'ENSG00000110395', 'ENSG00000111262', 'ENSG00000273079',
                         'ENSG00000133703', 'ENSG00000110844', 'ENSG00000111252', 'ENSG00000179295', 'ENSG00000083642', 'ENSG00000136158', 'ENSG00000176165', 'ENSG00000119596',
                         'ENSG00000179364', 'ENSG00000140262', 'ENSG00000169032', 'ENSG00000137834', 'ENSG00000080603', 'ENSG00000102962', 'ENSG00000174231', 'ENSG00000141510',
                         'ENSG00000196712', 'ENSG00000178691', 'ENSG00000136450', 'ENSG00000170836', 'ENSG00000141376', 'ENSG00000087191', 'ENSG00000161547', 'ENSG00000101752',
                         'ENSG00000152217', 'ENSG00000141646', 'ENSG00000134440', 'ENSG00000256463', 'ENSG00000105204', 'ENSG00000263002', 'ENSG00000160007', 'ENSG00000087088',
                         'ENSG00000063244', 'ENSG00000115758', 'ENSG00000119772', 'ENSG00000213639', 'ENSG00000198369', 'ENSG00000136710', 'ENSG00000115524', 'ENSG00000204217',
                         'ENSG00000036257', 'ENSG00000101266', 'ENSG00000171456', 'ENSG00000244462', 'ENSG00000124233', 'ENSG00000156304', 'ENSG00000159216', 'ENSG00000160201',
                         'ENSG00000099949', 'ENSG00000254709', 'ENSG00000183765', 'ENSG00000099995', 'ENSG00000100393', 'ENSG00000172936', 'ENSG00000168036', 'ENSG00000181555',
                         'ENSG00000169855', 'ENSG00000118007', 'ENSG00000155903', 'ENSG00000159692', 'ENSG00000068078', 'ENSG00000168769', 'ENSG00000113163', 'ENSG00000185129',
                         'ENSG00000145907', 'ENSG00000181163', 'ENSG00000165671', 'ENSG00000204469', 'ENSG00000204435', 'ENSG00000112640', 'ENSG00000171467', 'ENSG00000146247',
                         'ENSG00000164494', 'ENSG00000049618', 'ENSG00000065883', 'ENSG00000146830', 'ENSG00000005483', 'ENSG00000064419', 'ENSG00000157764', 'ENSG00000158941',
                         'ENSG00000164754', 'ENSG00000169249', 'ENSG00000215301', 'ENSG00000126752', 'ENSG00000072501', 'ENSG00000177485', 'ENSG00000101972', 'ENSG00000156531', 
                         'ENSG00000165509', 'ENSG00000167548'])
    
    df_LoF = df_LoF_orig[~(df_LoF_orig['gene_id'].isin(CE_genes)) & (df_LoF_orig[ac_col]<=kmax)].reset_index(drop = True)

    # Get LoFs in form a dictionary with LoF dataframe as values and Ensembl gene_id as item keys
    gene_dfs = construct_per_gene_dfs(df_LoF)

    # Predicted theoretical SFSs for all Roulette values and 500 selection values from (-10**-6 to -100) + neutral (s = 0) for allele counts from 0 to kmax
    with open(SFS_file, "rb") as f:
        SFSs = pickle.load(f)
    
    s_values = -np.logspace(np.log10(10**-6), np.log10(100), num=500)  # Grid of selection values. SFSs 2nd index num+1 (grid + neutral) values. Here it is 500+1 values

    # Fix probability of misannotation by taking the max value of each shet optimized likelihood

    # Initial parameters
    pnull_dict = {"stop_gained": 0.038656417410349544, "splice_donor_variant": 0.08395214833839681, "splice_acceptor_variant": 0.1022848249350011}
    
    stop_gain = pnull_dict["stop_gained"]
    splice_donor = pnull_dict["splice_donor_variant"]
    splice_acceptor = pnull_dict["splice_acceptor_variant"]
    initial_params_pnull = [stop_gain, splice_donor, splice_acceptor]

    # Bounds of parameters
    bound_pnull = [(1e-10, 0.5)] * len(initial_params_pnull)

    # Find the pnull which maximizes the likelihood of observing LoF SFS over all genes 
    result_pnull = minimize(
            p_misannot_s,
            initial_params_pnull,
            args=(gene_dfs, SFSs, ac_col),
            bounds=bound_pnull,
            method="Nelder-Mead",
            options={"disp": True}
        )
    
    pnull_dict["stop_gained"] = result_pnull.x[0]
    pnull_dict["splice_donor_variant"] = result_pnull.x[1]
    pnull_dict["splice_acceptor_variant"] = result_pnull.x[2]

    # Store pnull in outdir_pnull
    os.makedirs(outdir_pnull, exist_ok=True)
    pd.Series(pnull_dict, name="pnull").to_csv(f"{outdir_pnull}/{demography}.csv")
    # pnull results
    print("\nFinal Estimates:")
    print(f"pnull: {pnull_dict}")

    # Find the likelihood of observing LoF SFS as a function of s for all genes and store them
    pnull_dict = pd.read_csv(f"{outdir_pnull}/{demography}.csv", index_col=0)["pnull"].to_dict()

    LL_dict = {}
    
    # Prepare input list for parallel run for calculating likelihoods
    jobs = [
        delayed(gene_specific_LL_full)(
            curr_gene_id,
            gene_dfs[curr_gene_id],
            SFSs,
            pnull_dict,
            ac_col
        )
        for curr_gene_id in gene_dfs.keys()
    ]

    # Run in parallel (n_jobs = number of cores you want to use)
    results = Parallel(n_jobs=-1)(jobs)

    # Collect results
    for curr_gene_id, s_value in results:
        LL_dict[curr_gene_id] = s_value

    pkl_filename = pkl_LL_folder +f"LL_{demography}.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(LL_dict, f)
    print(f"Calculated likelihood for {demography} demography") 



def get_LL_estimates(demographies, LL_dict_folder, CES = False):
    """
    Load per-gene log-likelihoods for each ancestry and sum them across ancestries.

    Parameters
    ----------
    demographies : list of str
        e.g. ["NFE", "SAS", "FIN", "AFR", "AMR", "EAS"]
    LL_dict_folder : str
        Path prefix where LL_{demography}.pkl files are present
    CES : True if the likelihoods are for CES analysis. In that case, it is a 2D likelihood

    Returns
    -------
    LL_sum_dict : dict
        {gene_id: summed_log10_LL_vector}
    """
    LL_sum_dict = {}

    for demography in demographies:
        if CES == True:
            LL_dict_file = os.path.join(LL_dict_folder, f"LL_{demography}_CES.pkl")
        else:
            LL_dict_file = os.path.join(LL_dict_folder, f"LL_{demography}.pkl")

        with open(LL_dict_file, "rb") as f:
            LL_dict = pickle.load(f)

        for gene_id, LL_vec in LL_dict.items():
            LL_vec = np.asarray(LL_vec, dtype=float)

            if gene_id not in LL_sum_dict:
                LL_sum_dict[gene_id] = LL_vec.copy()
            else:
                # Sanity check: grids must match
                if LL_sum_dict[gene_id].shape != LL_vec.shape:
                    raise ValueError(
                        f"Shape mismatch for gene {gene_id} in {demography}"
                    )
                LL_sum_dict[gene_id] += LL_vec

    return LL_sum_dict



def log10sumexp(a, axis=None):
    """Stable log10(sum(10**a))"""
    return logsumexp(a * np.log(10), axis=axis) / np.log(10)

def per_gene_posterior_ests(LL_vec, s_vec, prior_mean, prior_SD, n_grid=1000):
    """
    LL_vec: log10-likelihood evaluated at s_vec
    s_vec : grid of the parameter (log10(s) values)
    prior_mean, prior_SD: mean/sd of the Normal prior on s_vec-scale
    Returns:
      integr_range: grid
      log10_post_density: log10 posterior density on grid (integrates to ~1)
    """
    s_vec = np.asarray(s_vec, dtype=float)
    LL_vec = np.asarray(LL_vec, dtype=float)

    # 1) grid for integration
    dx = s_vec[1] - s_vec[0]

    # 2) prior density on this scale
    prior_pdf = norm.pdf(s_vec, loc=prior_mean, scale=prior_SD)
    # avoid log10(0) underflow
    prior_pdf = np.maximum(prior_pdf, np.finfo(float).tiny)
    log10_prior = np.log10(prior_pdf)

    # 3) unnormalized log10 posterior density
    log10_post_unnorm = LL_vec + log10_prior

    # 4) normalize as a *density*: ∫ p(x) dx = 1  ->  sum p(x_i)*dx ≈ 1
    log10Z = log10sumexp(log10_post_unnorm + np.log10(dx))
    log10_post = log10_post_unnorm - log10Z

    return log10_post



def compute_LPs(LL_dict, gb_df, s_vec = -np.logspace(np.log10(10**-6), np.log10(100), num=500), SD = 0.5):
    """
    Computes the log posterior estimates for each gene based on the log-likelihoods
    and a specified prior distribution.
    """
    prior_dict = {gb_df["ensg"][i] : gb_df["prior_mean"][i] for i in range(len(gb_df))}

    s_vec = -s_vec
    # Assume s_vec is a NumPy array
    if np.any(s_vec <= 0):
        raise ValueError("Error: s_vec contains negative values.")
    
    if not np.all(np.diff(s_vec) > 0):
        raise ValueError("Error: s_vec is not sorted in ascending order.")
    
    s_vec = np.log10(s_vec)  # log10 transformed

    post_dict = {}
    for gene_id, LL_vec in LL_dict.items():
        if prior_dict.get(gene_id) is not None:
            prior_mean = np.log10(prior_dict[gene_id])
            prior_SD = SD
        else:
            prior_mean = -2
            prior_SD = 2

        Lpost_vals = per_gene_posterior_ests(LL_vec, s_vec, prior_mean, prior_SD)
        post_dict[gene_id] = {"Lpost_vals": Lpost_vals}

    return post_dict



def LL_to_dist(LL_vec, x_vec_log10, alphas=(0.1, 0.05)):
    """
    LL_vec: log10 unnormalized posterior/likelihood evaluated on x-grid
    x_vec_log10: x grid where x = log10(-s) (must be increasing)
    Returns summary stats for t = -s (positive) on linear scale.
    """
    x = np.asarray(x_vec_log10, dtype=float)
    LL = np.asarray(LL_vec, dtype=float)

    # Ensure increasing grid
    if np.any(np.diff(x) <= 0):
        raise ValueError("x_vec_log10 must be strictly increasing (sorted).")

    dx = x[1] - x[0]  # assumes uniform grid; if not uniform, handle separately

    # --- density in x-space: p(x) ∝ 10^LL(x)
    LL_shift = LL - np.nanmax(LL)  # do NOT modify input array
    w = 10.0 ** LL_shift           # unnormalized weights

    # Normalize as a *density* over x: sum p(x_i) dx = 1
    Z = np.sum(w) * dx
    p_x = w / Z                    # density in x

    # Transform to linear t = 10^x (= -s):
    t = 10.0 ** x
    # Jacobian: p_t(t) = p_x(x) / (t ln 10)
    p_t = p_x / (t * np.log(10.0))

    # Renormalize in t-space (should already be ~1, but do it for safety)
    mass = trapezoid(p_t, t)
    p_t = p_t / mass

    cdf = cumulative_trapezoid(p_t, t, initial=0.0)
    # cdf[-1] = 1.0  # guard numerical drift

    # Summary stats on t (= -s)
    t_mean = trapezoid(p_t * t, t)
    t_median = t[np.searchsorted(cdf, 0.5, side="left").clip(0, len(t)-1)]
    t_mode = t[np.argmax(p_t)]

    # Two-sided equal-tail credible intervals
    CIs = {}
    for alpha in alphas:
        lo_q = alpha / 2
        hi_q = 1 - alpha / 2
        lo = t[np.searchsorted(cdf, lo_q, side="left").clip(0, len(t)-1)]
        hi = t[np.searchsorted(cdf, hi_q, side="left").clip(0, len(t)-1)]
        CIs[alpha] = (lo, hi)

    return t_mean, t_median, t_mode, CIs


def infer_summary_stats(LL_dict, post_dict, s_val = -np.logspace(np.log10(10**-6), np.log10(100), num=500), alphas=(0.1, 0.05)):
    rows = []

    x_vec_log10 = np.log10(-s_val)
    
    for gene_id, LL_vec in LL_dict.items():
        L_mean, L_median, L_mode, L_CIs = LL_to_dist(LL_vec, x_vec_log10, alphas)

        post_vec = post_dict[gene_id]["Lpost_vals"]
        P_mean, P_median, P_mode, P_CIs = LL_to_dist(post_vec, x_vec_log10, alphas)

        row = {
            "gene_id": gene_id,
            "L_mean": L_mean,
            "L_median": L_median,
            "L_mode": L_mode,
            "Posterior_mean": P_mean,
            "Posterior_median": P_median,
            "Posterior_mode": P_mode,
        }
        for alpha in alphas:
            row[f"LL_CI{int(alpha*100)}_lower"] = L_CIs[alpha][0]
            row[f"LL_CI{int(alpha*100)}_upper"] = L_CIs[alpha][1]
            row[f"Posterior_CI{int(alpha*100)}_lower"] = P_CIs[alpha][0]
            row[f"Posterior_CI{int(alpha*100)}_upper"] = P_CIs[alpha][1]

        rows.append(row)

    print(f"{len(rows)} genes were summarized")
    return pd.DataFrame(rows)



def gene_specific_LL_CES_full(curr_gene_id, gene_df, Bell_tensor, pnull_dict, AC_col):
    
    num_kappa = Bell_tensor.shape[1] 
    num_s = Bell_tensor.shape[2] - 1 # exclude neutral 
    LL_surface = np.zeros((num_kappa, num_s), dtype=np.float64) 
    
    for annotation, pnull in pnull_dict.items(): 
        annotation_df = gene_df[gene_df["most_severe_consequence2"] == annotation].reset_index(drop=True) 
        if annotation_df.empty: 
            continue 
        mu_inds = annotation_df["MR_rank"].astype(int).values 
        ACs = annotation_df[AC_col].astype(int).values # Neutral component (last index) 
        p_neut = Bell_tensor[mu_inds, 0, -1, ACs] # shape (N, ) 
        # Selected components (all non-neutral) 
        p_sel = Bell_tensor[mu_inds, :, :-1, ACs] # shape (N, num_kappa, num_s) 
        
        # Combine with mixture # log10( pnull*p_neut + (1-pnull)*p_sel ) 
        mix = pnull * p_neut[:, None, None] + (1 - pnull) * p_sel 
        log_mix = np.log10(mix + 1e-300) # Sum over all variants (axis 0) 
        
        LL_surface += np.sum(log_mix, axis=0) 
        
    return curr_gene_id, LL_surface



def LoF_CES_likelihoods(LoF_filename, demography, SFS_folder, outdir_pnull, pkl_LL_folder, kmax = 1000):
    """
    Main function that calculates the likelihood(selection s) of observing SFS of LoFs
    Input:
        LoF_filename  :File containing a table of allele counts, allele number, mutation rate MR_rank, Ensembl gene_id, annotation (stop-gained, splice donor or splice acceptor) of LoFs. 
                       The allele counts and allele numbers are in columns, AC_{demography} and AN_{demography} respectively. 
                       MR_rank contains the row numbers of SFS hash table that we use in likelihood function.
                       annotation is stored in "most_severe_consequence2" column. 
        demography    : Demography for which we are calculating the likelihood
        SFS_folder      : contains SFS hashtable. (len(mut rate bins), len(selection grid), (ac = 0-kmax)) shape. See Make_SFS_demography.ipynb for more details.
        outdir_pnull  : Stores a csv file in outdir_pnull folder storing probability of misannotation values (pnull) for a demography.
    Output:
        pkl_LL_folder : Stores the log10(Likelihood) as function of s for all genes and given demography as a pickle file.
    """    

    # Get the LoF file loaded and get the columns that store allele counts
    df_LoF_orig = pd.read_csv(LoF_filename, sep="\t", compression="gzip")
    ac_col, AN_col = choose_ac_an_cols(demography)
    
    df_LoF = df_LoF_orig[(df_LoF_orig[ac_col]<=kmax)].reset_index(drop = True)

    # Get LoFs in form a dictionary with LoF dataframe as values and Ensembl gene_id as item keys
    gene_dfs = construct_per_gene_dfs(df_LoF)

    # Predicted theoretical SFSs for all Roulette values and 500 selection values from (-10**-6 to -100) + neutral (s = 0) for allele counts from 0 to kmax
    SFS_file = SFS_folder + f"/SFS_{demography}_CES.pkl"
    with open(SFS_file, "rb") as f:
        SFSs = pickle.load(f)

    kappa = np.logspace(0, 2, num = 50)
    s_values = -np.logspace(np.log10(1e-6), np.log10(100), num=100)  # Grid of selection values. SFSs 2nd index num+1 (grid + neutral) values. Here it is 500+1 values

    # Find the likelihood of observing LoF SFS as a function of s for all genes and store them
    pnull_dict = pd.read_csv(f"{outdir_pnull}/{demography}.csv", index_col=0)["pnull"].to_dict()

    LL_dict = {}
    
    # Prepare input list for parallel run for calculating likelihoods
    jobs = [
        delayed(gene_specific_LL_CES_full)(
            curr_gene_id,
            gene_dfs[curr_gene_id],
            SFSs,
            pnull_dict,
            ac_col
        )
        for curr_gene_id in gene_dfs.keys()
    ]

    # Run in parallel (n_jobs = number of cores you want to use)
    results = Parallel(n_jobs=-1)(jobs)

    # Collect results
    for curr_gene_id, s_value in results:
        LL_dict[curr_gene_id] = s_value

    pkl_filename = pkl_LL_folder +f"LL_{demography}_CES.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump(LL_dict, f)
    print(f"Calculated 2D likelihood for {demography} demography") 
    

















    



    