""" running the ntf over a range of frequencies."""
from forward import network_transfer_macrostable as nt
import numpy as np

def run_local_coupling_forward(C, D, params, freqs):
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

    for freq in freqs:
        w = 2 * np.pi * freq
        freq_model, freq_resp, eig_val, eig_vec = nt.network_transfer_local_alpha(
            C, D, params, w
        )
        frequency_response.append(freq_resp)
        eigenvalues.append(eig_val)
        eigenvectors.append(eig_vec)
        model_out.append(freq_model)

    frequency_response = np.asarray(frequency_response)
    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)
    model_out = np.transpose(np.asarray(model_out))

    return model_out, frequency_response, eigenvalues, eigenvectors

def run_local_coupling_forwards(brain, params, freqs):
    """run_forward. Function for running the forward model over the passed in range of frequencies,
    for the handed set of parameters (which must be passed in as a dictionary)

    Args:
        brain (Brain): An instance of the Brain class.
        params (dict): Dictionary of a setting of parameters for the NTF model.
        freqs (array): Array of freqencies for which the model is to be calculated.

    Returns:
        array: Model values for each frequency, for each region of the brain, ordered as according to HCP
        (as in Brain class ordering).

    """

    eigenvalues = []
    eigenvectors = []
    model_out = []

    ws = 2 * np.pi * freqs
    model_out, frequency_response, eigenvalues, eigenvectors = nt.network_transfer_local_alphas(
            brain, params, ws
        )

    model_out = np.transpose(np.asarray(model_out))

    return model_out, frequency_response, eigenvalues, eigenvectors
