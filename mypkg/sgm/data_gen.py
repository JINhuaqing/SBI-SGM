import torch
import numpy as np
from utils.reparam import logistic_torch
from utils.standardize import stdz_vec, psd_2tr

def simulator_data(params, sgmmodel, prior_bds):
    """The function to generate PSD and spatial feature
    """
    params_trans = []
    for param, prior_bd in zip(params, prior_bds):
        param_trans =  logistic_torch(param)*(prior_bd[1]-prior_bd[0]) + prior_bd[0]
        params_trans.append(param_trans)
    params_trans = torch.tensor(params_trans)
    
    psd, spatialFs = sgmmodel.run_local_coupling_forward(params_trans)
    psd = psd[:68, :]
    std_spatial = stdz_vec(spatialFs.sum(axis=1)) # std it
    
    std_psd_DB = psd_2tr(psd)
    out = np.concatenate([std_psd_DB.flatten(), std_spatial])
    return out