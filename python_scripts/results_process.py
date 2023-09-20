#!/usr/bin/env python
# coding: utf-8

# This file is to analyze the results and save them
# you should notice the _folder_path
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

import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# #### My own pkgs

# In[3]:


import importlib

from utils.misc import mag2db, meg_psd2spatialfeature, save_pkl_dict2folder, load_pkl_folder2dict, get_mode
from utils.standardize import stdz_psd, stdz_vec, minmax_vec, psd_2tr
from utils.brain_plot import U2BrainVec, reorderU 
from utils.reparam import theta_raw_2out, normcdf_np, normcdf_torch, logistic_np, logistic_torch
from utils.stable import paras_table_check
from sgm.sgm import SGM
from spectrome import Brain, path, functions
from features.psm import obt_psm_fs, obt_psm_raw
from constants import RES_ROOT, FIG_ROOT, DAT_ROOT
from models.embedding_nets import SummaryNet
#import sgm.data_gen 
#importlib.reload(sgm.data_gen)
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

paras.noise_sd = 1.6
paras.num_prior_sps = int(1e5)
paras.den_est = "nsf"
paras.is_embed = False

paras.prior_sd = 10


# In[7]:


# paras  for this file
_paras = edict()
_folder_path = f"./new_bds[0-9]*_reparam{paras.num_prior_sps:.0f}" +               f"_sd{paras.noise_sd*100:.0f}" +               f"_denest{paras.den_est}" +               f"_embed{paras.is_embed}_t3"
_paras.fils = list(RES_ROOT.glob(_folder_path))
_paras.fils = sorted(_paras.fils, key=lambda x: int(x.stem.split("bds")[1].split("_")[0]))


# In[8]:


prior = MultivariateNormal(loc=torch.zeros(7), covariance_matrix=torch.eye(7)*(paras.prior_sd**2))


# In[9]:


# obtain the brain
brain = Brain.Brain()
brain.add_connectome(DAT_ROOT)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()
sgmmodel = SGM(brain.reducedConnectome, brain.distance_matrix, FREQS)


# #### Some fns

# In[10]:


_map_fn_torch = partial(logistic_torch, k=0.1)
_theta_raw_2out = partial(theta_raw_2out, map_fn=partial(logistic_np, k=0.1))


# In[11]:


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


# In[ ]:





# ### Get posterior sampes
# (only need to run once)

# In[36]:


for cur_fil in _paras.fils[:]:
    cur_res = load_pkl_folder2dict(cur_fil, ["x", "*_raw"])
    post_spss_raw = []
    for ix in range(ind_psd.shape[-1]):
        curC = ind_conn[:, :, ix]
        curBrain = Brain.Brain()
        curBrain.add_connectome(DAT_ROOT) # grabs distance matrix
        # re-ordering for DK atlas and normalizing the connectomes:
        curBrain.reorder_connectome(curBrain.connectome, curBrain.distance_matrix)
        curBrain.connectome = curC# re-assign connectome to individual connectome
        curBrain.bi_symmetric_c()
        curBrain.reduce_extreme_dir()
            
        sp, raw_sps = meg_psd2spatialfeature(curBrain.reducedConnectome, ind_psd[:, :, ix], FREQS, band="alpha")
        
        std_spv = stdz_vec(raw_sps.sum(axis=1))
        std_psd_DB = psd_2tr(ind_psd[:, :, ix])
        
        curX_raw = np.concatenate([std_psd_DB.flatten(), std_spv])
        curX = torch.Tensor(curX_raw)
        post_sps_raw = cur_res.posterior.sample((10000, ), x=curX, max_sampling_batch_size=10000)
        post_sps_raw_stable = _filter_unstable(post_sps_raw, paras.prior_bds)
        print(post_sps_raw_stable.shape)
        post_spss_raw.append(post_sps_raw_stable.numpy())
            
            
    cur_res.post_spss_raw = post_spss_raw
    save_pkl_dict2folder(cur_fil, cur_res, is_force=False)
    
    # from theta_raw to theta
    post_spss = [_theta_raw_2out(post_sps_raw, paras.prior_bds) for post_sps_raw in cur_res.post_spss_raw]
    
    cur_res.post_spss = post_spss
    save_pkl_dict2folder(cur_fil, cur_res, is_force=False)


# In[ ]:





# ### point PSD estimate

# In[13]:


meg_psd_curve = np.mean([psd_2tr(ind_psd[:, :, ix]).mean(axis=0) for ix in range(36)], axis=0)
meg_psd_roi_sub = np.array([psd_2tr(ind_psd[:, :, ix]) for ix in range(36)])


# In[ ]:





# In[14]:


# the estimated PSD and spatial feature from SGM  

for cur_fil in _paras.fils:
    cur_res = load_pkl_folder2dict(cur_fil, ["poster*", "*_raw"])
    
    # way 1
    cur_est_parass = np.array([np.median(ps, axis=0) for ps in cur_res.post_spss])
    
    # way 2
    #cur_est_parass = np.array([[get_mode(ps[:, ix]) for ix in range(7)] for ps in cur_res.post_spss])
    
    # way 3
    #cur_est_parass_raw = np.array([np.median(ps, axis=0) for ps in cur_res.post_spss_raw])
    #cur_est_parass = _theta_raw_2out(cur_est_parass_raw, paras.prior_bds)
    
    # way 4
    #cur_est_parass_raw = np.array([[get_mode(ps[:, ix]) for ix in range(7)] for ps in cur_res.post_spss_raw])
    #cur_est_parass = _theta_raw_2out(cur_est_parass_raw, paras.prior_bds)
    
    std_psd_DB = []
    raw_sp = []
    for ix in trange(36):
        cur_est_paras = cur_est_parass[ix, :]
        
        cur_conn = ind_conn[:, :, ix]
        cur_brain = Brain.Brain()
        cur_brain.add_connectome(DAT_ROOT) # grabs distance matrix
        cur_brain.reorder_connectome(cur_brain.connectome, cur_brain.distance_matrix)
        cur_brain.connectome = cur_conn# re-assign connectome to individual connectome
        cur_brain.bi_symmetric_c()
        cur_brain.reduce_extreme_dir()
        
        cur_SGM_model = SGM(cur_brain.reducedConnectome, cur_brain.distance_matrix, FREQS)
        cur_psd, cur_sp = cur_SGM_model.run_local_coupling_forward(cur_est_paras)
        cur_psd = cur_psd[:68, :]
        cur_psd_DB = psd_2tr(cur_psd)
        
        std_psd_DB.append(cur_psd_DB)
        raw_sp.append(cur_sp.sum(axis=1))
        
    # the corr, sneak peek
    sgm_psd_curve = np.array(std_psd_DB).mean(axis=1).mean(axis=0)
    cor = scipy.stats.pearsonr(stdz_vec(meg_psd_curve), stdz_vec(sgm_psd_curve))[0]
    print(f"Corr is {cor:.3f}")    
    # Corrs for PSD and spatial features for each subject
    sgm_psd_roi_sub = np.array(std_psd_DB)
    corrs_roi_sub = []
    for ix in range(meg_psd_roi_sub.shape[0]):
        cur_meg = meg_psd_roi_sub[ix]
        cur_sgm = sgm_psd_roi_sub[ix]
        corrs_roi_sub.append(np.mean([scipy.stats.pearsonr(cur_meg[iy], cur_sgm[iy])[0] for iy in range(68)]))
    corrs_roi_sub = np.array(corrs_roi_sub)
    print(np.median(corrs_roi_sub), np.mean(corrs_roi_sub))
    
    # save the results
    cur_res.raw_sp = raw_sp
    cur_res.std_psd_DB = std_psd_DB
    save_pkl_dict2folder(cur_fil, cur_res, is_force=False)


# In[ ]:





# ### Full bayesian way

# In[ ]:


for cur_fil in _paras.fils:
    cur_res = load_pkl_folder2dict(cur_fil, ["raw_sp", "*_raw"])
    
    all_post_psds = []
    for ix in range(ind_psd.shape[-1]):
        curC = ind_conn[:, :, ix]
        curBrain = Brain.Brain()
        curBrain.add_connectome(DAT_ROOT) # grabs distance matrix
        # re-ordering for DK atlas and normalizing the connectomes:
        curBrain.reorder_connectome(curBrain.connectome, curBrain.distance_matrix)
        curBrain.connectome = curC# re-assign connectome to individual connectome
        curBrain.bi_symmetric_c()
        curBrain.reduce_extreme_dir()
        cur_sgmmodel = SGM(curBrain.reducedConnectome, curBrain.distance_matrix, FREQS)
            
        sp, raw_sps = meg_psd2spatialfeature(curBrain.reducedConnectome, ind_psd[:, :, ix], FREQS, band="alpha")
        
        std_spv = stdz_vec(raw_sps.sum(axis=1))
        std_psd_DB = psd_2tr(ind_psd[:, :, ix])
        
        curX_raw = np.concatenate([std_psd_DB.flatten(), std_spv])
        curX = torch.Tensor(curX_raw)
        cur_post = cur_res.posterior.set_default_x(curX)
            
        simulator_data_sp = partial(simulator_data,  sgmmodel=cur_sgmmodel, prior_bds=paras.prior_bds)
        simulator_data_wrapper, _ = prepare_for_sbi(simulator_data_sp, prior)
        
        tmp_sps, post_psd = simulate_for_sbi(simulator_data_wrapper, cur_post, 
                                num_simulations=1000,
                                num_workers=50)
        _, post_psd_stable = _filter_unstable(tmp_sps, paras.prior_bds, post_psd)
        print(ix, post_psd_stable.shape)
        all_post_psds.append(post_psd_stable.numpy())
        
    est_psds = []
    for post_psd in all_post_psds:
        est_psd = np.mean(post_psd[:, :-68], axis=0)
        est_psd = est_psd.reshape(68, 40)
        est_psds.append(est_psd)
        
    sgm_psd_curve_fb = np.array(est_psds).mean(axis=1).mean(axis=0)
    cor = scipy.stats.pearsonr(stdz_vec(meg_psd_curve), stdz_vec(sgm_psd_curve_fb))[0]
    print(f"Corr is {cor:.3f}")
    
    # Corrs for PSD and spatial features for each subject
    sgm_psd_roi_sub_fb = np.array(est_psds)
    corrs_roi_sub_fb = []
    for ix in range(meg_psd_roi_sub.shape[0]):
        cur_meg = meg_psd_roi_sub[ix]
        cur_sgm = sgm_psd_roi_sub_fb[ix]
        corrs_roi_sub_fb.append(np.mean([scipy.stats.pearsonr(cur_meg[iy], cur_sgm[iy])[0] for iy in range(68)]))
    corrs_roi_sub_fb = np.array(corrs_roi_sub_fb)
    
    print(np.median(corrs_roi_sub_fb), np.mean(corrs_roi_sub_fb))
    
    # save the results, including std PSD and std sp
    cur_res.fb_post_res = all_post_psds
    save_pkl_dict2folder(cur_fil, cur_res, is_force=False)


# In[ ]:




