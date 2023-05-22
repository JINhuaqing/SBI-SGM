#!/usr/bin/env python
# coding: utf-8

# This file is to try SBI_SGM for neuroimage R1. 
#only difference is I use t-3 error other than normal error
# 
# And I train model with a general connectome.
# 

# ### import and define

# In[1]:


import os
import sys
module_path = os.path.abspath(os.path.join('../mypkg'))
if module_path not in sys.path:
    sys.path.append(module_path)


# #### Other pkgs

# In[2]:


from tqdm import trange, tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from sbi import utils as sutils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import xarray as xr
from easydict import EasyDict as edict
from pathlib import Path
import pickle
import scipy

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from functools import partial
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SBI-SGM-mul')

parser.add_argument('--noise_sd', default=0.2, type=float, help='SD of error added to SGM output')
parser.add_argument('--num_prior_sps', default=int(1e5), type=int, help='Num of sps')
args = parser.parse_args()
# In[ ]:





# In[ ]:





# #### My own pkgs

# In[3]:


import importlib

import utils.misc
importlib.reload(utils.misc);
from utils.misc import mag2db, meg_psd2spatialfeature, save_pkl_dict2folder, load_pkl_folder2dict, get_mode

import utils.standardize
importlib.reload(utils.standardize);
from utils.standardize import stdz_psd, stdz_vec, minmax_vec, psd_2tr

import utils.brain_plot
importlib.reload(utils.brain_plot)
from utils.brain_plot import U2BrainVec, reorderU 

import utils.reparam
importlib.reload(utils.reparam)
from utils.reparam import theta_raw_2out, normcdf_np, normcdf_torch, logistic_np, logistic_torch

import utils.stable
importlib.reload(utils.stable)
from utils.stable import paras_table_check

import sgm.sgm
importlib.reload(sgm.sgm);
from sgm.sgm import SGM


from spectrome import Brain, path, functions

import features.psm
importlib.reload(features.psm)
from features.psm import obt_psm_fs, obt_psm_raw

import constants
importlib.reload(constants)
from constants import RES_ROOT, FIG_ROOT, DAT_ROOT

import models.embedding_nets 
importlib.reload(models.embedding_nets)
from models.embedding_nets import SummaryNet

import sgm.data_gen 
importlib.reload(sgm.data_gen)
from sgm.data_gen import simulator_data


# In[4]:


#torch.set_default_tensor_type(torch.DoubleTensor)


# ### Load real data PSD and brain

# #### PSD and connectome

# In[5]:


# Load data
ind_psd_xr = xr.open_dataarray(DAT_ROOT/'individual_psd_reordered_matlab.nc')
ind_psd = ind_psd_xr.values
FREQS = ind_psd_xr["frequencies"].values
ind_conn_xr = xr.open_dataarray(DAT_ROOT/'individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values


# #### Set up Parameters

# In[6]:


# some constant parameters for this file
paras = edict()
#bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (5, 20), (0.001,0.7), (0.001,2.0), (5.0,30.0))
#Bounds for taue, taui, alpha, speed, gei, gii, tauC
# (0.001,0.7), (0.001,2.0) when considering stable checking.
# (0.001,0.5), (0.001,1.5) as safe (narrower) bounds

# taue, taui, tauC, speed, alpha, gii, gei
paras.par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
paras.par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.names = ["Taue", "Taui", "TauC", "Speed", "alpha", "gii", "gei"]

paras.noise_sd = args.noise_sd
paras.num_prior_sps = args.num_prior_sps
paras.den_est = "nsf"
paras.is_embed = False
paras.num_rep = 10
paras.prior_sd = 10


# In[7]:


_map_fn_torch = partial(logistic_torch, k=0.1)
_theta_raw_2out = partial(theta_raw_2out, map_fn=partial(logistic_np, k=0.1))


# In[8]:


def _filter_unstable(theta_raw, prior_bds, x=None):
    """This fn is to remove unstable SGM parameters
        args: theta_raw: parameters: num of sps x dim
                order: ['Taue', 'Taui', 'TauC', 'Speed', 'alpha', 'gii', 'gei']
    """
    theta = _theta_raw_2out(theta_raw.numpy(), prior_bds)
    stable_idxs = paras_table_check(theta)
    
    # keep stable sps only
    theta_raw_stable = theta_raw[stable_idxs==0]
    if x is not None:
        x_stable = x[stable_idxs==0]
        return theta_raw_stable, x_stable
    else:
        return theta_raw_stable


# In[10]:


# paras  for this file
_paras = edict()
_folder_path = f"./new_bds_reparam{paras.num_prior_sps:.0f}" +               f"_sd{paras.noise_sd*100:.0f}" +               f"_denest{paras.den_est}" +               f"_embed{paras.is_embed}"
_paras.folder_path = RES_ROOT/_folder_path
print(f"folder is {_paras.folder_path}")
print(_paras.folder_path.exists())


# In[11]:




# ### Prepare for SBI

# #### Simulator

# In[12]:


def simulator(params, noise_sd, sgmmodel, prior_bds):
    """The function to simulate samples
    """
    params_trans = []
    for param, prior_bd in zip(params, prior_bds):
        param_trans =  _map_fn_torch(param)*(prior_bd[1]-prior_bd[0]) + prior_bd[0]
        params_trans.append(param_trans)
    params_trans = torch.tensor(params_trans)
    
    psd, spatialFs = sgmmodel.run_local_coupling_forward(params_trans)
    psd = psd[:68, :]
    std_spatial = stdz_vec(spatialFs.sum(axis=1)) # std it
    
    std_psd_DB = psd_2tr(psd)
    psd_fs = std_psd_DB.flatten()
    
    res = np.concatenate([psd_fs, std_spatial]) 
    noise = scipy.stats.t.rvs(3, size=res.shape);
    noise = noise_sd * noise/noise.std() # make sd to our pre-specified one
    
    return res+noise


# In[ ]:





# #### Prior 

# In[13]:


prior = MultivariateNormal(loc=torch.zeros(7), covariance_matrix=torch.eye(7)*(paras.prior_sd**2))


# #### Density estimator

# In[14]:


if paras.is_embed:
    embedding_net = SummaryNet(num_in_fs=68*41)
    paras.den_est = sutils.posterior_nn(
    model=paras.den_est, embedding_net=embedding_net)
    for ix in embedding_net.parameters():
        print(ix.sum(), ix.shape, ix.dtype)


# ### SBI

# In[15]:


# obtain the brain
brain = Brain.Brain()
brain.add_connectome(DAT_ROOT)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()
sgmmodel = SGM(brain.reducedConnectome, brain.distance_matrix, FREQS)

simulator_sp = partial(simulator,  noise_sd=paras.noise_sd, sgmmodel=sgmmodel, prior_bds=paras.prior_bds)
# make a SBI-wrapper on the simulator object for compatibility
simulator_wrapper, prior = prepare_for_sbi(simulator_sp, prior)


# In[ ]:





# #### RUN

# In[16]:


for iy in range(0, paras.num_rep):
    cur_res = edict()
    theta_raw, x = simulate_for_sbi(simulator_wrapper, prior, 
                                num_simulations=int(paras.num_prior_sps*1.1), # to make sure, after filtering unstable, we have enough sps
                                num_workers=50)
    
    
    # In[17]:
    
    
    # keep stable sps only
    theta_raw_stable, x_stable = _filter_unstable(theta_raw, paras.prior_bds, x)
    theta_raw_stable = theta_raw_stable[:int(paras.num_prior_sps), :]
    x_stable = x_stable[:int(paras.num_prior_sps), :]
    
    
    # In[18]:
    
    
    inference = SNPE(prior=prior, density_estimator=paras.den_est)
    density_estimator = inference.append_simulations(theta_raw_stable, x_stable).train()
    posterior = inference.build_posterior(density_estimator)
    
    
    # In[19]:
    
    
    cur_folder_path = f"./new_bds{iy+1}_reparam{paras.num_prior_sps:.0f}" + f"_sd{paras.noise_sd*100:.0f}" + f"_denest{paras.den_est}" + f"_embed{paras.is_embed}_t3"
    cur_folder_path = RES_ROOT/cur_folder_path
    cur_res.theta_raw = theta_raw_stable
    #cur_res.x = x_stable
    cur_res.posterior = posterior
    if paras.is_embed:
        cur_res.embedding_net = embedding_net
        for ix in embedding_net.parameters():
            print(ix.sum(), ix.shape)
    save_pkl_dict2folder(cur_folder_path, cur_res, is_force=False)


# In[ ]:





