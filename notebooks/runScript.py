#!/usr/bin/ipython
import os
import sys
import time
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from spectrome import Brain, path, functions
from tqdm import trange, tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sklearn.preprocessing import minmax_scale

import numpy as np
import xarray as xr
from easydict import EasyDict as edict
from pathlib import Path
import pickle
import scipy

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def mag2db(y):
    """Convert magnitude response to decibels for a simple array.

    Args:
        y (numpy array): Power spectrum, raw magnitude response.

    Returns:
        dby (numpy array): Power spectrum in dB

    """
    dby = 20 * np.log10(y)
    return dby

stdPSDFn = lambda x: (x- x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
stdFn = lambda x: (x-x.mean())/x.std()
minmaxFn = lambda x: (x-x.min())/(x.max()-x.min())



# Load data
ind_psd_xr = xr.open_dataarray('../data/individual_psd_reordered_matlab.nc')
ind_psd = ind_psd_xr.values
freqs = ind_psd_xr["frequencies"].values
ind_conn_xr = xr.open_dataarray('../data/individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values



data_dir = path.get_data_path()

# Get reorder indeix
# dk86.***.txt is the order for the MEG data
# DK68_labelorder.xlsx is the order for the template
dk_brainnet = pd.read_excel(Path(data_dir)/"DK68_labelorder.xlsx")
with open(Path(data_dir)/'dk86atlas_regions_python.txt') as f:
    dk_python = f.readlines()
    
brainnet_order = dk_brainnet["Order in Brainstorm"]
dk_python_renamed = []
brainnet_renamed = []

for i in range(len(dk_python)):
    sp1 = dk_python[i].split('=')
    sp2 = sp1[1].split('_')
    dk_python_renamed.append((sp2[0] + ' ' + sp2[1]).lower().strip())

for i in range(len(brainnet_order)):
    brainnet_renamed.append((brainnet_order[i].replace("'","")).lower())

d = {x: a for a, x in enumerate(dk_python_renamed[0:68])}
orderedind = np.array([d[x] for x in brainnet_renamed])


def reorderU(wU):
    wUreorder = wU.copy()
    wUreorder = wU[orderedind]
    return wUreorder

with open(Path(data_dir)/'BNVtemplate_DK68.txt', "r") as tf:
    DKtmplateRaw = tf.readlines()
DKtmplate = np.array([int(x.strip()) for x in DKtmplateRaw])

def U2BrainVec(wU):
    emVec = np.zeros_like(DKtmplate, dtype=np.float64)
    for idx in range(1, 69):
        emVec[DKtmplate==idx] = wU[idx-1]
    return emVec



paras = edict()
paras.noiseSd = 0.2
paras.numPriorSps = int(1e5)
paras.path = Path("./mainResults")

paras.robustParas = edict()
paras.robustParas.numRep = 20
paras.robustParas.noiseSd = 0.2
paras.robustParas.numPriorSps = int(1e5)
paras.robustParas.path = Path(paras.path/"robustResults")


if not paras.path.exists():
    paras.path.mkdir()
if not paras.robustParas.path.exists():
    paras.robustParas.path.mkdir()
    
    
def network_transfer_local_alpha(C, D, parameters, w):
    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF

    Returns:
        model_out (numpy asarray):  Each region's frequency response for
        the given frequency (w)
        frequency_response (numpy asarray):
        ev (numpy asarray): Eigen values
        Vv (numpy asarray): Eigen vectors

    """
    #C = brain.reducedConnectome
    #D = brain.distance_matrix

    
    parameters = np.asarray(parameters)
    tau_e = parameters[0]
    tau_i = parameters[1]
    tauC =  parameters[2]
    speed = parameters[3]
    alpha = parameters[4]
    gii =   parameters[5]  
    gei =   parameters[6]  
    gee = 1
    
    # Defining some other parameters used:
    zero_thr = 0.05

    # define sum of degrees for rows and columns for laplacian normalization
    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf

    nroi = C.shape[0]

    K = nroi

    Tau = 0.001 * D / speed
    Cc = C * np.exp(-1j * Tau * w)

    # Eigen Decomposition of Complex Laplacian Here
    L1 = np.identity(nroi)
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    L = L1 - alpha * np.matmul(np.diag(L2), Cc)

    d, v = np.linalg.eig(L)  
    eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
    eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
    eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

    eigenvalues = np.transpose(eig_val)
    eigenvectors = eig_vec[:, 0:K]

#     # Cortical model
    Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
    Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)
    FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)

    Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))
    
    Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

    Htotal = Hed + Hid


#     q1 = (1j * w + 1 / tau_e * Fe * eigenvalues)
    q1 = (1j * w + 1 / tauC * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response = np.divide(Htotal, q1)
    
    model_out = 0

    for k in range(K):
        model_out += (frequency_response[k]) * np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k])) 
    model_out2 = np.linalg.norm(model_out,axis=1)

    
    return model_out2, frequency_response, eigenvalues, eigenvectors


class sgm:
    def __init__(self, C, D, freqs, band="alpha"):
        self.freqs = freqs 
        self.C = C
        self.D = D
        if band == "alpha":
            self.freqband = np.where((freqs>=8) & (freqs<=12))[0]
        elif band == "beta":
            self.freqband = np.where((freqs>=13) & (freqs<=25))[0]
        else:
            self.freqband = np.arange(len(freqs))
            
    def run_local_coupling_forward(self, params):
        """run_forward. Function for running the forward model over the passed in range of frequencies,
        for the handed set of parameters (which must be passed in as a dictionary)
    
        Args:
            brain (Brain): An instance of the Brain class.
            #C = brain.reducedConnectome
            #D = brain.distance_matrix
            params (dict): Dictionary of a setting of parameters for the NTF model.
            freqs (array): Array of freqencies for which the model is to be calculated.
    
        Returns:
            array: Model values for each frequency, for each region of the brain, ordered as according to HCP
            (as in Brain class ordering).
    
        """
    
        eigenvalues = []
        eigenvectors = []
        frequency_response = []
        model_out = []
    
        for freq in self.freqs:
            w = 2 * np.pi * freq
            freq_model, freq_resp, eig_val, eig_vec = network_transfer_local_alpha(
                self.C, self.D, params, w
            )
            frequency_response.append(freq_resp)
            eigenvalues.append(eig_val)
            eigenvectors.append(eig_vec)
            model_out.append(freq_model)
    
        frequency_response = np.asarray(frequency_response)
        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors)
        model_out = np.transpose(np.asarray(model_out))
        
        model_out_band = model_out[:68, self.freqband]
        model_out_band_sum = np.sum(model_out_band,axis = 1)
        model_out_band_sum_norm = model_out_band_sum/np.linalg.norm(model_out_band_sum) # spatial feature on the specific band
    
        return model_out, model_out_band_sum_norm, frequency_response, eigenvalues, eigenvectors
    
# calculate the spatial feature for MEG under the given band
def MEGPSD2SF(C, PSD, freqs, band="alpha", is_More=False):
    if band == "alpha":
        freqband = np.where((freqs>=8) & (freqs<=12))[0]
    elif band == "beta":
        freqband = np.where((freqs>=13) & (freqs<=25))[0]
    else:
        freqband = np.arange(len(freqs))
    
    summed_PSD = np.sum(PSD[:,freqband], axis = 1)
    summed_PSD = summed_PSD/np.linalg.norm(summed_PSD)

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
    
    
    func1 = np.matmul(Cc2[0:68,0:68], summed_PSD)
    if is_More:
        return func1, summed_PSD
    else:
        return func1
    
    
def para2res(params):
    psd, spatialFs, _, _, _  = sgmmodel.run_local_coupling_forward(params)
    psdDB = mag2db(np.abs(psd))
    stdPsdDB = stdPSDFn(psdDB)
    stdSpatialFs = stdFn(spatialFs)
    return stdPsdDB, stdSpatialFs

def simulator(params):
    psd, sp = para2res(params)
    psd = psd[:68, :]
    #res = np.concatenate([psd.flatten(), sp])
    psdM = psd.mean(axis=0)
    res = np.concatenate([psdM, sp])
    noise =  np.random.randn(*res.shape)*paras.noiseSd
    return res + noise


#v_lower = 3.5-1.8
#v_upper = 3.5+1.8
#bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.5,10.0), (0.5,10.0), (5.0,30.0))
#This is the order for the above bounds:  tau_e, tau_i, alpha, speed, gei, gii, tauC,

# taue, taui, tauC, speed, alpha, gii, gei
par_low = np.asarray([0.005,0.005,0.005,1.7,0.1,0.5,0.5])
par_high = np.asarray([0.03, 0.20, 0.03,5.3,  1, 10, 10])


# obtain the brain
data_dir = path.get_data_path()
brain = Brain.Brain()
brain.add_connectome(data_dir)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()

sgmmodel = sgm(brain.reducedConnectome, brain.distance_matrix, freqs)


prior = utils.BoxUniform(low=torch.as_tensor(par_low), high=torch.as_tensor(par_high))
# make a SBI-wrapper on the simulator object for compatibility
simulator_wrapper, prior = prepare_for_sbi(simulator, prior)


for ix in range(paras.robustParas.numRep):
    theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=int(paras.robustParas.numPriorSps), num_workers=100)
    fname = f"./robustSps_{paras.robustParas.numPriorSps:.0f}_sd{paras.robustParas.noiseSd*100:.0f}_{ix+1}.pkl"
        
    inference = SNPE(prior=prior, density_estimator="mdn")
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    curRes = edict()
    curRes.theta = theta
    curRes.x = x
    curRes.posterior = posterior
    
    with open(paras.robustParas.path/fname, "wb") as f:
        pickle.dump(curRes, f)