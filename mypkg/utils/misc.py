import numpy as np
import scipy
import pickle
from easydict import EasyDict as edict


# load file from pkl
def load_pkl(fil):
    print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# load file from folder and save it to dict
def load_pkl_folder2dict(folder):
    res = edict()
    for fil in folder.glob('*.pkl'):
        res[fil.stem] = load_pkl(fil)
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force)

# save file to pkl
def save_pkl(fil, result, is_force=False):
    if not fil.parent.exists():
        fil.parent.mkdir()
        print(fil.parent)
        print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        print(f"{fil} exists! Use is_force=True to save it anyway")

# get the mode of a vec
def get_mode(x):
    kde_est = scipy.stats.gaussian_kde(x)
    xs = np.linspace(x.min(), x.max(), 500)
    ys = kde_est(xs)
    return xs[np.argmax(ys)]

def mag2db(y):
    """Convert magnitude response to decibels for a simple array.

    Args:
        y (numpy array): Power spectrum, raw magnitude response.

    Returns:
        dby (numpy array): Power spectrum in dB

    """
    dby = 20 * np.log10(y)
    return dby


# calculate the spatial feature for MEG under the given band
def meg_psd2spatialfeature(C, PSD, freqs, band="alpha"):
    """
        args:
            C: Connectome
            PSD: spectrum
            freqs: Frequency of the PSD
            band: the freq band selected
    """
    if band == "alpha":
        freqband = np.where((freqs>=8) & (freqs<=12))[0]
    elif band == "beta":
        freqband = np.where((freqs>=13) & (freqs<=25))[0]
    else:
        freqband = np.arange(len(freqs))
    
    summed_psd = np.sum(PSD[:,freqband], axis = 1)
    summed_psd = summed_psd/np.linalg.norm(summed_psd)

    w_spat = 10.0

    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    Cc = np.matmul(np.diag(L2), C)
    
    C2 = Cc + w_spat*np.eye(86)
    rowdegree = np.transpose(np.sum(C2, axis=1))
    coldegree = np.sum(C2, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf
    L22 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    Cc2 = np.matmul(np.diag(L22), C2)    
    
    func1 = np.matmul(Cc2[0:68,0:68], summed_psd)
    return func1, PSD[:, freqband] # first for analyse, second for plotting on brain Nrois x Nfreqs
