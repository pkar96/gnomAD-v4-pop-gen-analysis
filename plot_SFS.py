import pandas as pd
import numpy as np
import math
import scipy as sc
from scipy.integrate import simpson
from scipy import stats
import joblib
from joblib import Parallel, delayed, parallel_backend
import multiprocessing


def make_bins(jmin, bb, nn, incl_zero=True):
    """
    Construct integer bin edges for aggregating allele counts (SFS-style).

    Strategy
    -------
    - Start with unit-width bins [1,2), [2,3), … up to jmin.
    - Extend edges approximately geometrically by multiplying by `bb` until
      reaching ~nn/2.
    - Force final two edges to [ceil(nn/2), nn) so the last bin collects the
      upper half of the spectrum.
    - All bins are half-open: [left, right).

    Parameters
    ----------
    jmin : int
        Number of initial unit-width bins (≥1). Typical use: keep fine
        resolution in the rare-variant range.
    bb : float
        Growth factor (>1) for geometric expansion of bin edges after `jmin`.
    nn : int
        Sample size in allele counts (e.g., chromosomal count; 2N for diploids).
        Used only to position the terminal bins at ceil(nn/2) and nn.
    incl_zero : bool, default False
        If True, prepend 0 so monomorphic/reference counts can be binned in
        [0,1). If you're only analyzing segregating sites, leave False.

    Returns
    -------
    bins : np.ndarray of int, shape (K,)
        Strictly increasing integer edges from 1 (or 0) up to nn. Use with
        half-open tests (ac >= left) & (ac < right).

    Notes
    -----
    - Right edge is exclusive; ac == nn is not included (usually fine for SFS).
    - A minimal guard prevents duplicate edges that would create zero-width bins.
    """
    bins = np.arange(1, jmin + 1, dtype=int)
    bb_next = bb
    # catch up past starting bins
    while bins[-1] > bb_next:
        bb_next *= bb

    # choose integer edge; original rounding, then guard for strict increase
    bb_next = math.ceil(bb_next) if math.floor(bb_next) <= bins[-1] else math.floor(bb_next)
    if bb_next <= bins[-1]:
        bb_next = bins[-1] + 1
    bins = np.append(bins, bb_next)

    # keep expanding until ~nn/2
    while bins[-1] < nn/2:
        bb_next *= bb
        bb_next = math.ceil(bb_next) if math.floor(bb_next) == bins[-1] else math.floor(bb_next)
        if bb_next <= bins[-1]:
            bb_next = bins[-1] + 1
        bins = np.append(bins, bb_next)

    # pin the terminal region
    bins = np.concatenate((bins[:-1], [math.ceil(nn/2), nn]))
    if incl_zero:
        bins = np.concatenate(([0], bins))
    return bins


def bin_means(bins):
    """
    Mean integer allele-count value of each half-open bin [left, right).

    Parameters
    ----------
    bins : array-like of int
        Monotone integer bin edges.

    Returns
    -------
    means : np.ndarray of float, shape (len(bins)-1,)
        Arithmetic mean of the integers in each bin, i.e. of
        {left, left+1, ..., right-1}. Empty bins return NaN.

    Notes
    -----
    - Closed-form: mean of integers in [L, R) is (L + (R-1)) / 2 when R > L.
    - Identical numerical results to taking np.mean over np.arange(L, R).
    """
    bins = np.asarray(bins)
    L = bins[:-1].astype(float)
    R = bins[1:].astype(float)
    widths = R - L
    return np.where(widths > 0, (L + (R - 1.0)) / 2.0, np.nan)


def bin_sizes(bins):
    """
    Integer width of each bin.

    Parameters
    ----------
    bins : array-like of int
        Monotone integer bin edges.

    Returns
    -------
    widths : np.ndarray of float
        right - left for each adjacent pair (width in allele-count units).
    """
    bins = np.asarray(bins)
    return (bins[1:] - bins[:-1]).astype(float)


def bin_data(ac, nn, bins):
    """
    Sum values per bin given allele counts.

    Parameters
    ----------
    ac : np.ndarray of int
        Allele counts per site (same units as `bins`).
    nn : np.ndarray of float or int
        Per-site weights to aggregate (e.g., indicator 1 per site, or
        weights like coverage). Must be same shape as `ac`.
    bins : array-like of int
        Bin edges. Binning is half-open: [left, right); the rightmost edge
        is exclusive (ac == bins[-1] is excluded), matching the original code.

    Returns
    -------
    totals : np.ndarray of float
        Sum of `nn` for sites whose `ac` fall in each bin.

    Notes
    -----
    - Vectorized, result-identical replacement for the per-bin masking loop.
    - Uses np.searchsorted to find bin indices and np.bincount to accumulate.
    """
    bins = np.asarray(bins)
    ac = np.asarray(ac)
    nn = np.asarray(nn)

    out = np.zeros(len(bins) - 1, dtype=float)

    # Find bin index i such that bins[i] <= ac < bins[i+1] (half-open)
    idx = np.searchsorted(bins, ac, side='right') - 1
    valid = (idx >= 0) & (idx < len(bins) - 1)
    if np.any(valid):
        totals = np.bincount(idx[valid], weights=nn[valid], minlength=len(bins) - 1)
        out[:len(totals)] = totals
    return out


def plot_xy(ac, nn_tot, jmin, bb, incl_zero = True):

    """
    Calls above functions and returns allele counts and SFS.

    Parameters
    ----------
    ac : np.ndarray of int
        Allele counts per site (same units as `bins`).
    nn_tot : Allele number
    jmin : int
        Number of initial unit-width bins (≥1). Typical use: keep fine
        resolution in the rare-variant range.
    bb : float
        Growth factor (>1) for geometric expansion of bin edges after `jmin`.

    Returns
    -------
    means : np.ndarray of float, shape (len(bins)-1,)
        Arithmetic mean of the integers in each bin, i.e. of
        {left, left+1, ..., right-1}. Empty bins return NaN.
    denisty: np.ndarray of float, shape (len(bins)-1,)
        SFS values

    Notes
    -----
    - Vectorized, result-identical replacement for the per-bin masking loop.
    - Uses np.searchsorted to find bin indices and np.bincount to accumulate.
    """
    
    nn = np.ones_like(ac, dtype=float)         # each site counts once

    # Construct bins: unit bins up to 4, then geometric
    bins = make_bins(jmin=jmin, bb=bb, nn=nn_tot, incl_zero = incl_zero)
    # Bin sizes (widths in allele-count units)
    widths = bin_sizes(bins)
    # Raw totals per bin
    totals = bin_data(ac, nn, bins)
    # Normalize to density: counts per allele-count unit
    density = totals / widths
    # Optionally scale to probability mass = 1
    density = density / totals.sum()
    # Get bin midpoints
    means = bin_means(bins)

    return means, density
