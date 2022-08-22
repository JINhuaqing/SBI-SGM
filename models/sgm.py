from spectrome import functions

import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import pearsonr

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

np.random.seed(111)
tf.random.set_seed(111)

class sgm:
    def __init__(self, C_ind, D_ind, fvec):
        self._fvec = fvec
        self.connectome = C_ind
        self.distance_matrix = D_ind
    
    def simulate(self, parameters):

        def network_transfer(params,w):
            C = self.connectome
            D = self.distance_matrix
            tau_e = params[0]
            tau_i = params[1]
            tauC = params[2]
            speed = params[3]
            alpha = params[4]
            gii = params[5]
            gei = params[6]
            pw_scale = 1
            gee = 1

            zero_thr = 0.05
            # define sum of degrees for rows and columns for laplacian normalization
            rowdegree = np.transpose(np.sum(C, axis=1))
            coldegree = np.sum(C, axis=0)
            qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
            rowdegree[qind] = np.inf
            coldegree[qind] = np.inf

            nroi = C.shape[0]

            K= nroi
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

            Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))
            
            Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

            Htotal = Hed + Hid

            q1 = (1j * w + 1 / tauC * Fe * eigenvalues)
            qthr = zero_thr * np.abs(q1[:]).max()
            magq1 = np.maximum(np.abs(q1), qthr)
            angq1 = np.angle(q1)
            q1 = np.multiply(magq1, np.exp(1j * angq1))
            frequency_response = np.divide(Htotal, q1)
            
            model_out = 0

            for k in range(K):
                model_out += (frequency_response[k]) * np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k])) 
            model_out2 = np.linalg.norm(model_out,axis=1)
            
            return pw_scale*model_out2
        

        freq_mdl = []
        for freq in self._fvec:
            _w = 2 * np.pi * freq
            freq_model = network_transfer(parameters, _w)
            freq_mdl.append(freq_model)

        freq_mdl = np.transpose(np.asarray(freq_mdl))

        rois_with_MEG = np.arange(0,68)
        
        freq_mdl = freq_mdl[rois_with_MEG,:]

        freq_out = np.zeros(freq_mdl.shape)
        
        for p in np.arange(0,len(freq_mdl)):
            freq_out[p,:] = functions.mag2db(np.abs(freq_mdl[p,:]))
        
        return freq_out


# # # # # # # # # Probabilistic Model # # # # # # # # # # #    
class Probabilistic_SGM(tf.keras.Model):
    def __init__(self, fvec, connectome, distancematrix, IC, name = None):
        super(Probabilistic_SGM, self).__init__(name=name)
        self.SGM = sgm(C_ind = connectome, D_ind = distancematrix, fvec=fvec)
        self.taue_shape = tf.Variable(tf.random.normal([1], mean = IC[0]), name = 'tau_e_shape')
        self.taue_scale = tf.Variable(tf.random.normal([1]), name = 'tau_e_scale')
        self.taui_shape = tf.Variable(tf.random.normal([1], mean = IC[1]), name = 'tau_i_shape')
        self.taui_scale = tf.Variable(tf.random.normal([1]), name = 'tau_i_scale')
        self.tauC_shape = tf.Variable(tf.random.normal([1], mean = IC[2]), name = 'tauC_shape')
        self.tauC_scale = tf.Variable(tf.random.normal([1]), name = 'tauC_scale')
        self.v_shape = tf.Variable(tf.random.normal([1], mean = IC[3]), name = 'v_shape')
        self.v_scale = tf.Variable(tf.random.normal([1]), name = 'v_scale')
        self.alpha_shape = tf.Variable(tf.random.normal([1], mean = IC[4]), name = 'alpha_shape')
        self.alpha_scale = tf.Variable(tf.random.normal([1]), name = 'alpha_scale')
        self.gii_shape = tf.Variable(tf.random.normal([1], mean = IC[5]), name = 'gii_shape')
        self.gii_scale = tf.Variable(tf.random.normal([1]), name = 'gii_scale')
        self.gei_shape = tf.Variable(tf.random.normal([1], mean = IC[6]), name = 'gei_shape')
        self.gei_scale = tf.Variable(tf.random.normal([1]), name = 'gei_scale')
#         self.p_shape = tf.Variable(tf.random.normal([1], mean = 1e4), name = "p_shape")
#         self.p_scale = tf.Variable(tf.random.normal([1], mean=1e3), name = "p_scale")


    @property
    def qtaue(self):
        """Variational posterior for tau_e"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.te_shape), rate = tf.nn.softplus(self.te_scale))
        return tfd.Normal(tf.nn.softplus(self.taue_shape), tf.nn.softplus(self.taue_scale))
    
    @property
    def qtaui(self):
        """Variational posterior for tau_i"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.ti_shape), rate = tf.nn.softplus(self.ti_scale))
        return tfd.Normal(tf.nn.softplus(self.taui_shape), tf.nn.softplus(self.taui_scale))

    @property
    def qtauC(self):
        """Variational posterior for tauC"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c1_shape), rate = tf.nn.softplus(self.c1_scale))
        return tfd.Normal(tf.nn.softplus(self.tauC_shape), tf.nn.softplus(self.tauC_scale))

    @property
    def qv(self):
        """Variational posterior for v"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c2_shape), rate = tf.nn.softplus(self.c2_scale))
        return tfd.Normal(tf.nn.softplus(self.v_shape), tf.nn.softplus(self.v_scale))

    @property
    def qalpha(self):
        """Variational posterior for alpha"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c3_shape), rate = tf.nn.softplus(self.c3_scale))
        return tfd.Normal(tf.nn.softplus(self.alpha_shape), tf.nn.softplus(self.alpha_scale))

    @property
    def qgii(self):
        """Variational posterior for gii"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c4_shape), rate = tf.nn.softplus(self.c4_scale))
        return tfd.Normal(tf.nn.softplus(self.gii_shape), tf.nn.softplus(self.gii_scale))
    
    @property
    def qgei(self):
        """Variational posterior for gei"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.ae_shape), rate = tf.nn.softplus(self.ae_scale))
        return tfd.Normal(tf.nn.softplus(self.gei_shape), tf.nn.softplus(self.gei_scale))

#     @property
#     def qp(self):
#         """Variational posterior for p"""
#         # return tfd.Gamma(concentration = tf.nn.softplus(self.ae_shape), rate = tf.nn.softplus(self.ae_scale))
#         return tfd.Normal(tf.nn.softplus(self.p_shape), tf.nn.softplus(self.p_scale))

    @property
    def qstd(self):
        """Variational posterior for sigma"""
        # return tfd.LogNormal(loc = tf.nn.softplus(self.sigma_loc), scale = tf.nn.softplus(self.sigma_scale))
        # return tfd.Normal(tf.nn.softplus(self.sigma_loc), scale = tf.nn.softplus(self.sigma_scale))
        # return tfd.InverseGamma(tf.exp(self.sigma_loc), tf.exp(self.sigma_scale))
        return tfd.InverseGamma(tf.exp(tf.random.normal([1])), tf.exp(tf.random.normal([1])))
        # return tfd.InverseGamma(tf.nn.softplus(tf.random.normal([1])), tf.nn.softplus(tf.random.normal([1])))


    def call(self, obs, sampling=True):
        """Predict p(y|theta)?"""
        sample = lambda x: x.sample().numpy() if sampling else x.mean()
        parameters = np.array([
                                sample(self.qtaue),
                                sample(self.qtaui),
                                sample(self.qtauC),
                                sample(self.qv),
                                sample(self.qalpha),
                                sample(self.qgii),
                                sample(self.qgei)
#                                 sample(self.qp),
                                ])

        # Forward model simulation
        fullspectrum = self.SGM.simulate(parameters)
        std = tf.sqrt(sample(self.qstd))
#         loc = fullspectrum.astype('float32')
        loc = minmax_scale(fullspectrum.T).astype('float32')
        density = tfd.Normal(loc, std)
        # print(tf.reduce_mean(density.log_prob(obs)))
        return density.log_prob(obs)
#         return -pearsonr(obs, loc)[0].astype('float32')


    @property
    def losses(self):
        """Sum of KL divergences between posterior and priors"""
        prior_taue = tfd.Normal(0.010,0.008)
        prior_taui = tfd.Normal(0.020,0.010)
        prior_tauC = tfd.Normal(0.020,0.010)
        prior_v = tfd.Normal(15,10)
        prior_alpha = tfd.Normal(0.5,0.4)
        prior_gii = tfd.Normal(1,0.8)
        prior_gei = tfd.Normal(1,0.8)
#         prior_p = tfd.Normal(1e4,1e3)
        # prior_std = tfd.ExpInverseGamma(0.5, 0.5)
        # prior_std = tfd.LogNormal(loc = 0.5, scale = 0.5)
        return (tf.reduce_sum(tfd.kl_divergence(self.qtaue, prior_taue)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qtaui, prior_taui)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qtauC, prior_tauC)) +
                tf.reduce_sum(tfd.kl_divergence(self.qv, prior_v)) +
                tf.reduce_sum(tfd.kl_divergence(self.qalpha, prior_alpha)) +
                tf.reduce_sum(tfd.kl_divergence(self.qgii, prior_gii)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qgei, prior_gei))) 
#                 + 
#                 tf.reduce_sum(tfd.kl_divergence(self.qp, prior_p)))
