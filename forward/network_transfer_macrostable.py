"""Module for computing basic quantities from a spectral graph model: the forward model
Makes the calculation for a single frequency only. """

import numpy as np

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

    tau_e = parameters["tau_e"]
    tau_i = parameters["tau_i"]
    speed = parameters["speed"]
    gei = parameters[
        "gei"
    ]  # excitatory-inhibitory synaptic conductance as ratio of E-E syn
    gii = parameters[
        "gii"
    ]  # inhibitory-inhibitory synaptic conductance as ratio of E-E syn
    tauC = parameters["tauC"]
    alpha = parameters["alpha"]
#     gee = parameters["gee"]
    
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
    print(Tau)
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


def network_transfer_local_alphas(brain, parameters, ws):
    """Network Transfer Function for spectral graph model.

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF

    Returns:
        model_out (numpy asarray):  Each region's frequency response for
        the given frequency (ws)
        frequency_response (numpy asarray):
        ev (numpy asarray): Eigen values
        Vv (numpy asarray): Eigen vectors

    """
    C = brain.reducedConnectome
    D = brain.distance_matrix
    
    tau_e = parameters["tau_e"]
    tau_i = parameters["tau_i"]
    speed = parameters["speed"]
    gei = parameters[ "gei" ]  # excitatory-inhibitory synaptic conductance as ratio of E-E syn
    gii = parameters[ "gii" ]  # inhibitory-inhibitory synaptic conductance as ratio of E-E syn
    tauC = parameters["tauC"]
    alpha = parameters["alpha"]
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
    Cc = C[:, :, np.newaxis] * np.exp(-1j * Tau[:, :, np.newaxis] * ws) # d x d x m
    
    # Eigen Decomposition of Complex Laplacian Here
    L1 = np.identity(nroi)
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    Ls = L1 - alpha * np.matmul(np.diag(L2), np.transpose(Cc, [2, 0, 1])) # m x d x d
    
    #
    eigenvaluess = []
    eigenvectorss = []
    for L in Ls:
        d, v = np.linalg.eig(L)  
        eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
        eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
        eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index
    
        eigenvalues = np.transpose(eig_val)
        eigenvectors = eig_vec[:, 0:K]
        eigenvaluess.append(eigenvalues)
        eigenvectorss.append(eigenvectors)
    eigenvaluess = np.array(eigenvaluess) # m x d
    eigenvectorss = np.array(eigenvectorss) # m x d x d
    
      # Cortical model
    Fe = np.divide(1 / tau_e ** 2, (1j * ws + 1 / tau_e) ** 2)
    Fi = np.divide(1 / tau_i ** 2, (1j * ws + 1 / tau_i) ** 2)
    FG = np.divide(1 / tauC ** 2, (1j * ws + 1 / tauC) ** 2)
    
    Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * ws + Fi * gii/tau_i)))/(1j * ws + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * ws + Fi * gii / tau_i)))
    
    Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * ws + Fe * gee/tau_e)))/(1j * ws + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * ws + Fe * gee / tau_e)))
    Htotals = Hed + Hid
    
    
    q1s = (1j * ws[:, np.newaxis] + 1 / tauC * FG[:, np.newaxis] * eigenvaluess)
    qthrs = zero_thr * np.abs(q1s[:, :]).max(axis=-1)
    magq1s =  np.maximum(np.abs(q1s[:, :]), qthrs[:, np.newaxis])
    angq1s = np.angle(q1s)
    q1s = np.multiply(magq1s, np.exp(1j * angq1s))
    frequency_responses = np.divide(Htotals[:, np.newaxis], q1s)
    
    model_outs = frequency_responses[:, np.newaxis, np.newaxis, :] *  eigenvectorss[:, :, np.newaxis, :] * np.conjugate(eigenvectorss[:, np.newaxis, :, :])
    model_outs = model_outs.sum(axis=-1)
    model_outs2 = np.linalg.norm(model_outs,axis=2)
    return model_outs2, frequency_responses, eigenvaluess, eigenvectorss