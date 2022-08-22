from spectrome import functions

import numpy as np

class sgm_local:
    def __init__(self, fvec):
        self._fvec = fvec

    
    def simulate(self, parameters):

        def network_transfer(params,w):
            tau_e = params[0]
            tau_i = params[1]
            gii = params[2]
            gei = params[3]
            # pw_scale = params[4]
            pw_scale = 1
            gee = 1


            # Cortical model
            Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
            Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)

            Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))
            
            Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

            Htotal = Hed + Hid
            
            return pw_scale*Htotal
        

        freq_mdl = []
        for freq in self._fvec:
            _w = 2 * np.pi * freq
            freq_model = network_transfer(parameters, _w)
            freq_mdl.append(freq_model)

        freq_mdl = np.transpose(np.asarray(freq_mdl))
        
        # freq_out = functions.mag2db(np.abs(freq_mdl))
        
        return np.abs(freq_mdl)
