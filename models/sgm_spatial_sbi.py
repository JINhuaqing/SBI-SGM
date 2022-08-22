from spectrome import functions

import numpy as np
from sklearn.preprocessing import minmax_scale

class sgm:
    def __init__(self, C_ind, D_ind, fvec):
        self._fvec = fvec
        self.connectome = C_ind
        self.distance_matrix = D_ind
    
    def simulate(self, parameters):

        def network_transfer(params,w):
            C = self.connectome
            D = self.distance_matrix
            params = np.asarray(params)
            tau_e = params[0]
            tau_i = params[1]
            tauC = params[2]
            speed = params[3]
            alpha = params[4]
            gii = params[5]
            # gee = params[6]
            gei = params[6]
            pw_scale = 1
            # gei = 1
            gee = 1

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

            d, v = np.linalg.eig(L)  # decomposition with scipy.linalg.eig
            eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
            eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
            eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

            eigenvalues = np.transpose(eig_val)
            eigenvectors = eig_vec[:, 0:K]

            # Cortical model
            Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
            Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)
            # FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)

            Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))
            
            Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

            Htotal = Hed + Hid

            q1 = (1j * w + 1 / tauC * Fe * eigenvalues)
            # q1 = (1j * w + 1 / tauC * FG * eigenvalues)
            qthr = zero_thr * np.abs(q1[:]).max()
            magq1 = np.maximum(np.abs(q1), qthr)
            angq1 = np.angle(q1)
            q1 = np.multiply(magq1, np.exp(1j * angq1))
            frequency_response = np.divide(Htotal, q1)
            
            model_out = 0

            for k in range(K):
                # model_out += (frequency_response[k]) * np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k])) 
                model_out += frequency_response[k] * eigenvectors[:, k] 
            # model_out2 = np.linalg.norm(model_out,axis=1)
            model_out2 = np.abs(model_out)
            
            rois_with_MEG = np.arange(0,68)
            Xk = np.empty((len(rois_with_MEG),K))

            for k in range(K):
                model_out = np.linalg.norm(((frequency_response[k]) * np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k]))), axis=1)
                Xk[:,k] = model_out[rois_with_MEG]
            
            return pw_scale*model_out2, Xk
        
        rois_with_MEG = np.arange(0,68)
        freq_mdl = []
        freqband = np.where((self._fvec>=8) & (self._fvec<=12))[0]
        eigvec_ns = np.zeros((len(rois_with_MEG),86,len(self._fvec)))
        
        for i, freq in enumerate(self._fvec):
            _w = 2 * np.pi * freq
            freq_model, eigmodes = network_transfer(parameters, _w)
            freq_mdl.append(freq_model)
            eigvec_ns[:,:,i] = eigmodes

        freq_mdl = np.transpose(np.asarray(freq_mdl))
        eigvec_ns_summed = np.sum(eigvec_ns[:,:,freqband],axis = 2)
        
        freq_mdl = freq_mdl[rois_with_MEG,:]

        freq_out = np.zeros(freq_mdl.shape)
        
        for p in np.arange(0,len(freq_mdl)):
            freq_out[p,:] = np.abs(freq_mdl[p,:])
            # freq_out[p,:] = functions.mag2db(np.abs(freq_mdl[p,:]))
        
        return freq_out, eigvec_ns_summed
