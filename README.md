# gnomAD-v4-pop-gen-analysis
Contains codes to generate estimates and plots in "_Inference of elevated mutation rates and variant effects using 700k exomes_", Kar _et al._, bioRxiv, 2026.

The datasets used are large and have not been uploaded but we will update with the upload soon.

The repository contains codes to 

1. Get mutation rate distributions - Mutation_rate_estimation folder
2. Recent demography estimates - Demography_estimation
Once these parameters are obtained, generate a hash table of the predicted SFS (allele counts < 5000) for mutation rate, kappa (only for CES), demography and selection parameters.
This can be done using Make_SFS_demography.ipynb for LoF, Make_SFS_demography_CES.ipynb for finding CES genes, Make_SFS_demography_missense.ipynb for missense analysis.
Applications
1. LoF estimates can be found by running the codes in LoF_selection
2. CES genes are also found in LoF_selection (LoF_CES_analysis.ipynb).
3. Missense estimates can be found using the code in Missense folder
   Once the predicted SFS are made using Make_SFS_demography_missense.ipynb and stored in sfs.pkl as a pickle file, they can be used to generate missense selection estimates.
   Just run Generate_missense.run_pipeline

Example 
score_col = "AlphaMissense"
# input file contains the missense protein language model score in score_col, mutation rate model (eg. - Roulette), allele count from pop gen dataset, Ensembl gene id
input_file = "../Data/missense/df_missense_AC.txt.gz"
# file where new estimates are stored
output_file = f"/n/data2/hms/dbmi/sunyaev/lab/pkar/Missense_analysis/final/{score_col}.pkl"
# need LoF scores to resolve duplicates. Should contain Ensembl gene id, LoF constraint score (higher score -> higher constraint)
lof_ref_path = "../LoF_selection/LoF_s_het.txt.gz"

# MLE params (optimization parameters)
init_c = 0.1
init_beta = 1
init_log_sigma = np.log(0.5551)
bounds = [(-4, 4), (0, 10), (np.log(dx), np.log(10.0))]

# Run the code to get the estimates
_,_,_ = Generate_missense.run_pipeline(input_file, output_file, score_col, allele_count_max=5000,
    sfs_mle_pkl="/n/data2/hms/dbmi/sunyaev/lab/pkar/demography_SFS/sfs.pkl",
    sfs_variant_pkl="/n/data2/hms/dbmi/sunyaev/lab/pkar/demography_SFS/sfs.pkl",
    # MLE params
    init_c=init_c,
    init_beta=init_beta,
    init_log_sigma=init_log_sigma,
    bounds=bounds,
    maxiter=400,
    fatol=1e-8,
    # parallel
    n_jobs=-1,
    backend="multiprocessing",
    verbose=10,
    # outputs
    gene_mle_out_dir=None,
    # optional LoF dedup
    lof_ref_path=lof_ref_path,
    lof_ref_score_col="Posterior_CI10_lower",
    key_cols=None)
