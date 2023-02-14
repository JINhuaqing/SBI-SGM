#!/usr/bin/env python
# coding: utf-8

# This file is to try SBI_SGM when extracting features with fooof (power spectrum model)
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

import warnings
warnings.filterwarnings("ignore")


# In[ ]:





# #### My own pkgs

# In[3]:


import importlib

import utils.misc
importlib.reload(utils.misc);
from utils.misc import mag2db, meg_psd2spatialfeature, save_pkl_dict2folder, load_pkl_folder2dict

import utils.standardize
importlib.reload(utils.standardize);
from utils.standardize import stdz_psd, stdz_vec, minmax_vec, psd_2tr

import utils.reparam
importlib.reload(utils.reparam)
from utils.reparam import logistic_np, logistic_torch, theta_raw_2out

import sgm.sgm
importlib.reload(sgm.sgm);
from sgm.sgm import SGM


from spectrome import Brain, path, functions

import features.psm
importlib.reload(features.psm)
from features.psm import obt_psm_fs

import constants
importlib.reload(constants)
from constants import RES_ROOT, FIG_ROOT, DAT_ROOT

import models.embedding_nets 
importlib.reload(models.embedding_nets)
from models.embedding_nets import SummaryNet


# #### Some fns for this file only

# In[4]:


def _remove_nopeak(theta, x):
    """The function is to remove the samples with no-peak case
        args:
            theta: the parameters, n x nparas
            x: the samples, n x dim of features
    """
    keep_idxs = np.abs(x).min(axis=1).values != 0
    #keep_idxs = x.max(axis=1).values != np.inf
    x = x[keep_idxs, :]
    theta = theta[keep_idxs, :]
    return theta, x


# In[5]:


def _stdpsd_2psmfs(std_psd_DB):
    """ convert std_psd_DB to psm features
    """
    std_psd_power = 10**(std_psd_DB) # lets change it to exp scale
    psd_psm_fs = []
    for ix in range(68):
        cur_psd = std_psd_power[ix, :]
        cur_fs = obt_psm_fs(cur_psd, FREQS)
        psd_psm_fs.append(cur_fs)
    psd_psm_fs = np.concatenate(psd_psm_fs)
    return psd_psm_fs


# In[ ]:





# ### Load real data PSD and brain

# #### PSD and connectome

# In[6]:


# Load data
ind_psd_xr = xr.open_dataarray(DAT_ROOT/'individual_psd_reordered_matlab.nc')
ind_psd = ind_psd_xr.values
FREQS = ind_psd_xr["frequencies"].values
ind_conn_xr = xr.open_dataarray(DAT_ROOT/'individual_connectomes_reordered.nc')
ind_conn = ind_conn_xr.values


# #### Set up Parameters

# In[7]:


# some constant parameters for this file
paras = edict()
#v_lower = 3.5-1.8
#v_upper = 3.5+1.8
#bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.5,10.0), (0.5,10.0), (5.0,30.0))
#This is the order for the above bounds:  tau_e, tau_i, alpha, speed, gei, gii, tauC,

# taue, taui, tauC, speed, alpha, gii, gei
paras.par_low = np.asarray([0.005,0.005,0.005,1.7,0.1,0.5,0.5])
paras.par_high = np.asarray([0.03, 0.20, 0.03,5.3,  1, 10, 10])
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.names = ["Taue", "Taui", "TauC", "Speed", "alpha", "gii", "gei"]

# paras 
paras.noise_sd = 0.20
paras.num_prior_sps = int(1.8e3)
paras.den_est = "nsf"
paras.num_round = 5
paras.is_embed = True


# In[8]:


# paras  for this file
_paras = edict()
_folder_path = f"./fooofMRReparam_{paras.num_prior_sps:.0f}" +               f"_sd{paras.noise_sd*100:.0f}" +               f"_denest{paras.den_est}" +               f"_numround{paras.num_round:.0f}" +               f"_embed{paras.is_embed}"
_paras.folder_path = RES_ROOT/_folder_path
print(f"folder is {_paras.folder_path}")
print(_paras.folder_path.exists())


# In[9]:


cur_res = edict()


# In[ ]:





# ### Prepare for SBI

# #### Simulator

# In[10]:


def simulator(params, noise_sd, sgmmodel, prior_bds):
    """The function to simulate samples
    """
    params_trans = []
    for param, prior_bd in zip(params, prior_bds):
        param_trans =  logistic_torch(param)*(prior_bd[1]-prior_bd[0]) + prior_bd[0]
        params_trans.append(param_trans)
    params_trans = torch.tensor(params_trans)
    
    psd, spatialFs = sgmmodel.run_local_coupling_forward(params_trans)
    psd = psd[:68, :] # only keep first 68
    std_spatial = stdz_vec(spatialFs.sum(axis=1)) # std it
    
    std_psd_DB = psd_2tr(psd)
    noise1 =  np.random.randn(*std_psd_DB.shape)*noise_sd 
    std_psd_DB_noise = std_psd_DB + noise1
    psd_psm_fs = _stdpsd_2psmfs(std_psd_DB_noise) # psd psm fs, two peaks
    psd_mean_fs = stdz_vec(std_psd_DB_noise.mean(axis=0)) # mean PSD, 40 dim
    
    noise2 =  np.random.randn(*std_spatial.shape)*noise_sd 
    res = np.concatenate([psd_psm_fs, psd_mean_fs, std_spatial+noise2]) 
    return res


# #### Prior 

# In[11]:


#prior = sutils.BoxUniform(low=torch.as_tensor(paras.par_low), high=torch.as_tensor(paras.par_high))
prior = MultivariateNormal(loc=torch.zeros(7), covariance_matrix=torch.eye(7)*100)


# In[ ]:





# ### SBI

# #### RUN

# In[ ]:


# obtain the brain
#for ix in range(19, 31):
for ix in range(33, ind_psd.shape[-1]):
    curC = ind_conn[:, :, ix]
    curBrain = Brain.Brain()
    curBrain.add_connectome(DAT_ROOT) # grabs distance matrix
    # re-ordering for DK atlas and normalizing the connectomes:
    curBrain.reorder_connectome(curBrain.connectome, curBrain.distance_matrix)
    curBrain.connectome = curC# re-assign connectome to individual connectome
    curBrain.bi_symmetric_c()
    curBrain.reduce_extreme_dir()
        
    # the simulator
    sgmmodel = SGM(curBrain.reducedConnectome, curBrain.distance_matrix, FREQS)
    simulator_sp = partial(simulator,  
                           noise_sd=paras.noise_sd, 
                           sgmmodel=sgmmodel,
                           prior_bds=paras.prior_bds)
    # make a SBI-wrapper on the simulator object for compatibility
    simulator_wrapper, prior = prepare_for_sbi(simulator_sp, prior)
    
    # the observed data
    sp, raw_sps = meg_psd2spatialfeature(curBrain.reducedConnectome, 
                                         ind_psd[:, :, ix], FREQS, 
                                         band="alpha")
    std_spv = stdz_vec(raw_sps.sum(axis=1))
    std_psd_DB = psd_2tr(ind_psd[:, :, ix])
    psm_fs = _stdpsd_2psmfs(std_psd_DB)
    psd_mean_fs = stdz_vec(std_psd_DB.mean(axis=0)) # mean PSD, 40 dim
    
    curX_raw = np.concatenate([psm_fs, psd_mean_fs, std_spv])
    curX = torch.Tensor(curX_raw)


    if paras.is_embed:
        # embedding net
        cur_embedding_net = SummaryNet(num_in_fs=516)
        cur_den_est = sutils.posterior_nn(
        model=paras.den_est, embedding_net=cur_embedding_net)
        #for iy in embedding_net.parameters():
        #    print(iy.sum(), iy.shape)
    else:
        cur_den_est = paras.den_est
    
    inference = SNPE(prior=prior, 
                     density_estimator=cur_den_est)
    proposal = prior 
    for _ in range(paras.num_round):
        theta, x = simulate_for_sbi(simulator_wrapper, proposal,
                                    num_simulations=paras.num_prior_sps, 
                                    num_workers=50)
        theta, x = _remove_nopeak(theta, x) # remove some samples
        density_estimator = inference.append_simulations(
                            theta, x, proposal=proposal
                            ).train()
        posterior = inference.build_posterior(density_estimator)
        
        
        #update proposal 
        proposal = posterior.set_default_x(curX)
        
    cur_res[f"posterior_{ix+1}"] = posterior
    cur_res[f"posteriorfix_{ix+1}"] = proposal
    if paras.is_embed:
        cur_res[f"embedding_net_{ix+1}"] = cur_embedding_net
        del cur_embedding_net
    save_pkl_dict2folder(_paras.folder_path, cur_res)


# In[ ]:





# #### Get posterior

# In[97]:


cur_res = load_pkl_folder2dict(_paras.folder_path)


# In[48]:


post_spss_raw = []
for ix in range(ind_psd.shape[-1]):
    post_sps_raw = cur_res[f"posteriorfix_{ix+1}"].sample((10000, ), max_sampling_batch_size=100000)
    post_spss_raw.append(post_sps_raw.numpy())
        


# In[50]:


cur_res.post_spss_raw = post_spss_raw
save_pkl_dict2folder(_paras.folder_path, cur_res)


# In[51]:


# In[ ]:


# from theta_raw to theta
post_spss = [theta_raw_2out(post_sps_raw, paras.prior_bds) for post_sps_raw in cur_res.post_spss_raw]

cur_res.post_spss = post_spss
save_pkl_dict2folder(_paras.folder_path, cur_res, True)




# In[ ]:





# #### PSD estimate

# In[52]:


cur_res = load_pkl_folder2dict(_paras.folder_path)


# In[53]:


# the estimated PSD and spatial feature from SGM  
# way 1
cur_est_parass = np.array([np.median(ps, axis=0) for ps in cur_res.post_spss])

# way 2
#cur_est_parass = np.array([[get_mode(ps[:, ix]) for ix in range(7)] for ps in cur_res.post_spss])

# way 3
#cur_est_parass_raw = np.array([np.median(ps, axis=0) for ps in cur_res.post_spss_raw])
#cur_est_parass = theta_raw_2out(cur_est_parass_raw, paras.prior_bds)

# way 4
#cur_est_parass_raw = np.array([[get_mode(ps[:, ix]) for ix in range(7)] for ps in cur_res.post_spss_raw])
#cur_est_parass = theta_raw_2out(cur_est_parass_raw, paras.prior_bds)

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
    


# In[54]:


sgm_psd_curve = np.array(std_psd_DB).mean(axis=1).mean(axis=0)
meg_psd_curve = np.mean([psd_2tr(ind_psd[:, :, ix]).mean(axis=0) for ix in range(36)], axis=0)
cor = scipy.stats.pearsonr(stdz_vec(meg_psd_curve), stdz_vec(sgm_psd_curve))[0]
print(f"Corr is {cor:.3f}")

