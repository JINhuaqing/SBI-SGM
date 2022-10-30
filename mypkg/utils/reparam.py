# this file contains fns for reparameterizing SGM
import numpy as np
import torch


def logistic_torch(x, k=0.05):
    """k=0.2 fits prior N(0, 100)
    """
    num = torch.exp(k*x)
    den = torch.exp(k*x) + 1
    
    # fix inf issue
    res = num/den
    res[torch.isinf(num)] = 1
    return res


def logistic_np(x, k=0.05):
    """k=0.2 fits prior N(0, 100)
    """
    num = np.exp(k*x)
    den = np.exp(k*x) + 1
    # fix inf issue
    res = num/den
    res[np.isinf(num)] = 1
    return res

def theta_raw_2out(thetas_raw, prior_bds):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x 7
              prior_bds: an array with 7 x 2
    """
    assert prior_bds.shape[0] == 7
    assert thetas_raw.shape[1] == 7
    thetas_out = logistic_np(thetas_raw)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
    return thetas_out