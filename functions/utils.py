from mne import connectivity
import numpy as np
import pandas as pd
from scipy.signal import firls
from spectrum import pmtm

def label_meg(MEG, labels):
    # Permutation for 68 cortex regions, JuliaOrder from OLD ASHISH MATLAB CODE
    MEGPerm = np.concatenate(
        [np.arange(5), np.arange(6,9), np.arange(10,16), np.array([17, 16]), np.arange(18,32), np.array([5, 32, 33, 9])]
    )
    MEG_cortex_perm = np.concatenate([MEGPerm, MEGPerm + 34])
    # Recreate labeled dataframe AFTER permutation
    for i, val in enumerate(labels.dropna().values):
        short_name = (val[0].replace('.','-'))
        labels.loc[i, 'region_used'] = 'ctx-' + short_name

    MEG = MEG[MEG_cortex_perm, :]
    template_ordered_labels = np.squeeze(labels.reindex(MEG_cortex_perm).values)
    return pd.DataFrame(MEG.T, columns = template_ordered_labels)

def sort_connectome(connectivity_matrix, distance_matrix):
    """Change connectome region ordering to match standard ordering.
    DEPRECATED

    Args:
        connectivity_matrix ([type]): [description]
        distance_matrix ([type]): Distance adjacency matrix in millimeters

    Returns:
        Cij: Connectivity matrix reordered
        Dij: Distance adjacency matrix, reordered and converted to meters
    """
    # permutation to match region indexes with time series
    cortexstart = 18
    permutation = np.concatenate(
        [np.arange(cortexstart, 86), np.arange(0, cortexstart)]
    )
    Cij = connectivity_matrix[permutation,][:,permutation]
    Dij = distance_matrix[permutation,][:,permutation]
    # normalization for row sum of connectivity matrix
    Cij = Cij / (np.sum(Cij, 1) * np.ones([68,68]))
    return Cij, Dij/1000

def normalize_connectome(connectivity_matrix, distance_matrix):
    """Normalize for row sum of connectivity and convert mm to m for distance"""
    return connectivity_matrix / (np.sum(connectivity_matrix, 1) * np.ones_like(connectivity_matrix)), distance_matrix

def get_multitaper(timecourse, nfft, eigen, tapers):
    Sk_complex, weights, _ = pmtm(timecourse, NFFT=nfft, e=eigen, v=tapers, show=False)
    Sk = np.abs(Sk_complex) ** 2
    Sk = np.mean(Sk.T * weights, axis = 1)
    return Sk[:int(nfft/2)]

def get_frequencies(nfft, fs, fmin, fmax):
    bins = np.linspace(1, nfft, nfft)
    fbin = bins/nfft * fs 
    freq = fbin[:int(nfft/2)]
    band = np.where(np.logical_and(freq >= fmin, freq <= fmax))
    return np.squeeze(band), freq[np.squeeze(band)]

def get_bandpassfilter(fs, fmin = 2, fmax = 45):
    gains = np.array([0, 0, 1, 1, 0, 0])
    return firls(73, np.array([0, 0.2*fmin, 0.9*fmin, fmax, fmax+10, fmax+30]), desired = gains, fs = fs)