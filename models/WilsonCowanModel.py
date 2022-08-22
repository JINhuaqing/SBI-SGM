from functions.utils import get_multitaper, get_frequencies
import numpy as np
from scipy.integrate import odeint
from ddeint import ddeint
from spectrum import dpss
from mne.connectivity import spectral_connectivity
from sklearn.preprocessing import minmax_scale
from scipy.stats import pearsonr

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# from fooof import FOOOF
np.random.seed(111)
tf.random.set_seed(111)

class WilsonCowanModel:
    def __init__(self, times, nfft, fmin = 2, fmax = 45, y0 = None):
        self._times = times
        # self._y0 = np.random.randn(2)
        self._y0 = np.array([0.1, 0.1]) 
        self._nfft = nfft
        self.fs = int(np.round(1000/(times[1] - times[0])))
        self.fmin = fmin
        self.fmax = fmax
        bins  = np.linspace(1, nfft, nfft)
        # NDPSS = len(times[self.fs:])
        NDPSS = len(times)
        self.findices, self.frequencies = get_frequencies(nfft, self.fs, fmin=fmin, fmax=fmax)
        [self._tapers, self._eigen] = dpss(NDPSS, 3)
        # self.hbp = firls(101, np.array([0, 0.2*self.fmin, 0.9*self.fmin, self.fmax, self.fmax+10, 75]), desired = np.array([0, 0, 1, 1, 0, 0]), fs = self.fs)
        # self.lpf = np.array([1, 2, 5, 2, 1])/np.sum([1, 2, 5, 2, 1])

    def _simulate(self, parameters, times):
        # set up some default parameters we don't need to optimize:
        # _y0 = np.random.randn(2)
        tau_e, tau_i, c1, c2, c3, c4, kE, kI, P = [float(x) for x in parameters]
        rE = 1
        rI = 1
        # P  = 2.25
        Q  = 0

        def sigmoid(x, a, thr):
            """Response/sigmoidal functions expressing the nonlinearity of the interactions

            Args:
                x ([type]): [description]
                a ([type]): Parameter alpha
                thr ([type]): Parameter theta

            Returns:
                [type]: [description]
            """
            return 1/(1 + np.exp(-a * (x - thr)))

        def Se(x):
            aE = 1.3
            thrE = 4
            return sigmoid(x, aE, thrE) - sigmoid(0, aE, thrE)

        def Si(x):
            aI = 2
            thrI = 3.7
            return sigmoid(x, aI, thrI) - sigmoid(0, aI, thrI)

        def WilsonCowan(y, t, p):
            """Wilson Cowan neural oscillator model

            Args:
                y ([type]): variable for current solution in integrator
                t (float): variable for current time step in integrator
                p (list) - Parameters:
                tau_e ([type]): [description]
                tau_i ([type]): [description]
                c1 ([type]): Coupling term for c_EE
                c2 ([type]): Coupling term for c_EI
                c3 ([type]): Coupling term for c_II
                c4 ([type]): Coupling term for c_IE
                rE ([type]): Excitatory refractory period
                rI ([type]): Inhibitory refractory period
                P ([type]): External input/current injection to excitatory population.
                Q ([type]): External input/current injection to inhibitory population.

            Returns:
                [E(t), I(t)]: Array, solution to Wilson-Cowan ODE.
            """
            E, I = y
            y1 = (-E + (kE - rE * E) * Se(c1 * E - c2 * I + P))/tau_e
            y2 = (-I + (kI - rI * I) * Si(c3 * E - c4 * I + Q))/tau_i
            return [y1, y2]

        sol = odeint(WilsonCowan, self._y0, times, (parameters,))
        excit_sol, inhib_sol = sol.T
        return excit_sol - inhib_sol

    def _compute_pmtm(self, timecourse):
        return get_multitaper(timecourse, self._nfft, self._eigen, self._tapers)

    def _fit_periodic(self, psd):
        fm = FOOOF()
        fm.add_data(self.freqs, psd, [self.fmin, self.fmax])
        fm.fit()
        return fm._peak_fit

    def simulate(self, x):
        """Simulate periodic power spectrum

        Args:
            x (array): array of model parameters
        Returns:
            ymodel: model simulated timecourse
            full_spectrum
            periodic_spectrum
        """
        ymodel = self._simulate(x, self._times)
        # compute power spectrum, after 1 second stabilizing period
        # full_spectrum = self._compute_pmtm(ymodel[self.fs:])[self.frange]
        full_spectrum = self._compute_pmtm(ymodel)[self.findices]
        # periodic_spectrum = self._fit_periodic(full_spectrum)
        # self._ymodel = ymodel[self.fs:]
        self._ymodel = ymodel
        self._full_spectrum = full_spectrum
        # self._periodic_spectrum = periodic_spectrum
        # if np.any(np.isnan(periodic_spectrum)):
        #     print('NaNs occured at: {}'.format(x))
        return ymodel, full_spectrum


# # # # # # # Network Model # # # # # # #
class WilsonCowanNetwork:
    def __init__(self, times, Cjk, Djk, fmin = 2, fmax = 45, y0 = None):
        self._times = times
        self.fs = int(np.round(1000/(times[1] - times[0])))
        self._history = lambda t: 0.1*np.random.randn(1, 68*2)[0]
        # self._history = lambda t: np.ones([1, 68*2])[0] * np.random.randn(1)
        self._Cjk = Cjk    # connectivity
        self._Djk = Djk    # distance matrix
        self.fmin = fmin
        self.fmax = fmax
    
    def _simulate(self, parameters, times):
        tau_e, tau_i, c1, c2, c3, c4, c5, P, kE, kI = [float(x) for x in parameters]
        rE = 1
        rI = 1
        # P  = np.ones([68]) * 1.25
        # P[57] = 1.25
        # P  = 1.25 
        Q  = 0

        def sigmoid(x, a, thr):
            """Response/sigmoidal functions expressing the nonlinearity of the interactions

            Args:
                x ([type]): [description]
                a ([type]): Parameter alpha
                thr ([type]): Parameter theta

            Returns:
                [type]: [description]
            """
            return 1/(1 + np.exp(-a * (x - thr)))

        def Se(x):
            aE = 1.3
            thrE = 4
            return sigmoid(x, aE, thrE) - sigmoid(0, aE, thrE)

        def Si(x):
            aI = 2
            thrI = 3.7
            return sigmoid(x, aI, thrI) - sigmoid(0, aI, thrI)

        def compute_delays(D, speed = 10):
            """assuming meters per second uniform transmission speed between all regions"""
            return D/speed 

        def WilsonCowanNet(y, t, tau_e, tau_i, c1, c2, c3, c4, c5, P, Cjk, Djk):
            """Wilson Cowan Network Modeel

            Args:
                y ([type]): variable for current solution in integrator
                t (float): variable for current time step in integrator
                p (list) - Parameters:
                tau_e ([type]): [description]
                tau_i ([type]): [description]
                c1 ([type]): Coupling term for c_EE
                c2 ([type]): Coupling term for c_EI
                c3 ([type]): Coupling term for c_II
                c4 ([type]): Coupling term for c_IE
                rE ([type]): Excitatory refractory period
                rI ([type]): Inhibitory refractory period
                P ([type]): External input/current injection to excitatory population.
                Q ([type]): External input/current injection to inhibitory population.

            Returns:
                [E(t), I(t)]: Array, solution to Wilson-Cowan ODE.
            """
            E = y(t)[:68]    # 68 cortical region always
            I = y(t)[68:]    # second half is inhibitory population    
            y1 = np.zeros(68)
            y2 = np.zeros(68)
            # for each region, grab the region specific connectome rows/columns and compute
            for j in np.arange(0,68):
                Cj = Cjk[j,:]   # j-th region's connectivity strengths
                Dj = Djk[j,:]   # j-th region's delay terms
                # Connectome term:
                conn_term = np.dot(Cj,  y(t-Dj.all())[0:68])
                # Network model
                y1[j] = (-E[j] + (kE - rE * E[j]) * Se(c1 * E[j] - c2 * I[j] + c5 * conn_term + P))/tau_e
                y2[j] = (-I[j] + (kI - rI * I[j]) * Si(c3 * E[j] - c4 * I[j] + Q))/tau_i

            return np.concatenate((y1, y2), axis = None)

        self._delays = compute_delays(self._Djk)
        solnet = ddeint(WilsonCowanNet, self._history, self._times, fargs = (tau_e, tau_i, c1, c2, c3, c4, c5, P, self._Cjk, self._delays))
        return np.transpose(solnet[:,:68] - solnet[:,68:])

    def _coherence_connectivity(self, fwd):
        """Compute Coherence connectivity with multitaper spectrtum

        Args:
            fwd (array): Model simulated data
        Returns:
            con (array): Nregion x Nregion connectivity with only lower triangle indices filled
        """
        # alpha band:
        FLB = 8
        FUB = 13
        con, _, _, _, _ = spectral_connectivity(
            [fwd],
            method = 'coh',
            sfreq = self.fs,
            mode = 'multitaper',
            fmin = FLB,
            fmax= FUB,
            faverage = True,
            mt_adaptive = True,
            mt_low_bias = False,
            verbose = 'ERROR'
        )
        # only lower triangle matters (tril)
        return np.squeeze(con)

    def simulate(self, x):
        """Simulate model periodic spectrum with FOOOF

        Args:
            x (array): model parameters

        Returns:
            ymodel: simulated time course Nregions x Ntimes
            coh_fc: simulated coherence FC Nregions x Nregions
        """
        ymodel = self._simulate(x, self._times)
        coh_fc = self._coherence_connectivity(ymodel[:,self.fs:])
        self._ymodel = ymodel
        self._coh_fc = coh_fc
        return ymodel, coh_fc


# # # # # # # # # Probabilistic Model # # # # # # # # # # #
class Probabilistic_NMM(tf.keras.Model):
    def __init__(self, d, fs, nfft = 4096, name = None):
        super(Probabilistic_NMM, self).__init__(name=name)
        self.times = np.linspace(0, d, int(d*fs))
        self.NMM = WilsonCowanModel(self.times, nfft = nfft)
        # initialize parameters
        self.te_shape = tf.Variable(tf.random.normal([1], mean = 20), name = 'tau_e_shape')
        self.te_scale = tf.Variable(tf.random.normal([1]), name = 'tau_e_scale')
        self.ti_shape = tf.Variable(tf.random.normal([1], mean = 20), name = 'tau_i_shape')
        self.ti_scale = tf.Variable(tf.random.normal([1]), name = 'tau_i_scale')
        self.c1_shape = tf.Variable(tf.random.normal([1]), name = 'c1_shape')
        self.c1_scale = tf.Variable(tf.random.normal([1]), name = 'c1_scale')
        self.c2_shape = tf.Variable(tf.random.normal([1]), name = 'c2_shape')
        self.c2_scale = tf.Variable(tf.random.normal([1]), name = 'c2_scale')
        self.c3_shape = tf.Variable(tf.random.normal([1]), name = 'c3_shape')
        self.c3_scale = tf.Variable(tf.random.normal([1]), name = 'c3_scale')
        self.c4_shape = tf.Variable(tf.random.normal([1], mean = 15), name = 'c4_shape')
        self.c4_scale = tf.Variable(tf.random.normal([1]), name = 'c4_scale')
        self.ke_shape = tf.Variable(tf.random.normal([1], mean = 10), name = 'ke_shape')
        self.ke_scale = tf.Variable(tf.random.normal([1]), name = 'ke_scale')
        self.ki_shape = tf.Variable(tf.random.normal([1], mean = 10), name = 'ki_shape')
        self.ki_scale = tf.Variable(tf.random.normal([1]), name = 'ki_scale')
        self.p_shape = tf.Variable(tf.random.normal([1], mean = 5), name = "p_shape")
        self.p_scale = tf.Variable(tf.random.normal([1]), name = "p_scale")
        # self.sigma_loc = tf.Variable(tf.random.normal([1]), name = "sigma_loc")
        # self.sigma_scale = tf.Variable(tf.random.normal([1]), name = "sigma_scale")

    @property
    def qte(self):
        """Variational posterior for tau_e"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.te_shape), rate = tf.nn.softplus(self.te_scale))
        return tfd.Normal(tf.nn.softplus(self.te_shape), tf.nn.softplus(self.te_scale))
    
    @property
    def qti(self):
        """Variational posterior for tau_i"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.ti_shape), rate = tf.nn.softplus(self.ti_scale))
        return tfd.Normal(tf.nn.softplus(self.ti_shape), tf.nn.softplus(self.ti_scale))

    @property
    def qc1(self):
        """Variational posterior for c1"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c1_shape), rate = tf.nn.softplus(self.c1_scale))
        return tfd.Normal(tf.nn.softplus(self.c1_shape), tf.nn.softplus(self.c1_scale))

    @property
    def qc2(self):
        """Variational posterior for c2"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c2_shape), rate = tf.nn.softplus(self.c2_scale))
        return tfd.Normal(tf.nn.softplus(self.c2_shape), tf.nn.softplus(self.c2_scale))

    @property
    def qc3(self):
        """Variational posterior for c3"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c3_shape), rate = tf.nn.softplus(self.c3_scale))
        return tfd.Normal(tf.nn.softplus(self.c3_shape), tf.nn.softplus(self.c3_scale))

    @property
    def qc4(self):
        """Variational posterior for c4"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.c4_shape), rate = tf.nn.softplus(self.c4_scale))
        return tfd.Normal(tf.nn.softplus(self.c4_shape), tf.nn.softplus(self.c4_scale))
    
    @property
    def qke(self):
        """Variational posterior for kE"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.ae_shape), rate = tf.nn.softplus(self.ae_scale))
        return tfd.Normal(tf.nn.softplus(self.ke_shape), tf.nn.softplus(self.ke_scale))

    @property
    def qki(self):
        """Variational posterior for kE"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.ae_shape), rate = tf.nn.softplus(self.ae_scale))
        return tfd.Normal(tf.nn.softplus(self.ki_shape), tf.nn.softplus(self.ki_scale))

    @property
    def qpt(self):
        """Variational posterior for external excitatory input"""
        # return tfd.Gamma(concentration = tf.nn.softplus(self.p_shape), rate = tf.nn.softplus(self.p_scale))
        return tfd.Normal(tf.nn.softplus(self.p_shape), tf.nn.softplus(self.p_scale))

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
                                sample(self.qte),
                                sample(self.qti),
                                sample(self.qc1),
                                sample(self.qc2),
                                sample(self.qc3),
                                sample(self.qc4),
                                sample(self.qke),
                                sample(self.qki),
                                sample(self.qpt)
                                ])
        # if np.any(parameters < 0.05):
            # print(parameters)
            # return 0.0
        # else:
        # Forward model simulation
        _, fullspectrum = self.NMM.simulate(parameters)
        std = tf.sqrt(sample(self.qstd))
        loc = minmax_scale(fullspectrum).astype('float32')
        density = tfd.Normal(loc, std)
        # print(tf.reduce_mean(density.log_prob(obs)))
        return density.log_prob(obs)
        # return -pearsonr(obs, loc)[0].astype('float32')

    @property
    def losses(self):
        """Sum of KL divergences between posterior and priors"""
        # prior_ts = tfd.Gamma(10,1)
        # prior_c1 = tfd.Gamma(16,1)
        # prior_c2 = tfd.Gamma(12,1)
        # prior_c3 = tfd.Gamma(15,1)
        # prior_c4 = tfd.Gamma(3,1)
        # prior_ae = tfd.Gamma(1.3,1)
        # prior_ai = tfd.Gamma(2,1)
        # prior_thre = tfd.Gamma(4,1)
        # prior_thri = tfd.Gamma(4.7,1)
        # prior_pt = tfd.Gamma(1.5,1)
        prior_ts = tfd.Normal(10,1)
        prior_c1 = tfd.Normal(16,1)
        prior_c2 = tfd.Normal(12,1)
        prior_c3 = tfd.Normal(15,1)
        prior_c4 = tfd.Normal(3,1)
        prior_ks = tfd.Normal(1,1)
        prior_ae = tfd.Normal(1.3,1)
        prior_ai = tfd.Normal(2,1)
        prior_thre = tfd.Normal(4,1)
        prior_thri = tfd.Normal(4.7,1)
        prior_pt = tfd.Normal(1.25,1)
        # prior_std = tfd.ExpInverseGamma(0.5, 0.5)
        # prior_std = tfd.LogNormal(loc = 0.5, scale = 0.5)
        return (tf.reduce_sum(tfd.kl_divergence(self.qte, prior_ts)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qti, prior_ts)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qc1, prior_c1)) +
                tf.reduce_sum(tfd.kl_divergence(self.qc2, prior_c2)) +
                tf.reduce_sum(tfd.kl_divergence(self.qc3, prior_c3)) +
                tf.reduce_sum(tfd.kl_divergence(self.qc4, prior_c4)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qke, prior_ks)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qki, prior_ks)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qpt, prior_pt)))


class Probabilistic_NetworkNMM(tf.keras.Model):
    def __init__(self, tvec, C, D, name = None):
        super(Probabilistic_NetworkNMM, self).__init__(name=name)
        self.times = tvec
        self.NMM   = WilsonCowanNetwork(tvec, Cjk = C, Djk = D)
        # [self.tapers, self.eigen] = dpss(len(tvec[fs:]), 7)
        # self.nfft  = nfft
        # self._fs   = fs
        ################# Initialize Parameters ################
        self.te_shape = tf.Variable(tf.random.normal([1], mean = 20), name = 'tau_e_shape')
        self.te_scale = tf.Variable(tf.random.normal([1]), name = 'tau_e_scale')
        self.ti_shape = tf.Variable(tf.random.normal([1], mean = 20), name = 'tau_i_shape')
        self.ti_scale = tf.Variable(tf.random.normal([1]), name = 'tau_i_scale')
        self.c1_shape = tf.Variable(tf.random.normal([1]), name = 'c1_shape')
        self.c1_scale = tf.Variable(tf.random.normal([1]), name = 'c1_scale')
        self.c2_shape = tf.Variable(tf.random.normal([1]), name = 'c2_shape')
        self.c2_scale = tf.Variable(tf.random.normal([1]), name = 'c2_scale')
        self.c3_shape = tf.Variable(tf.random.normal([1]), name = 'c3_shape')
        self.c3_scale = tf.Variable(tf.random.normal([1]), name = 'c3_scale')
        self.c4_shape = tf.Variable(tf.random.normal([1], mean = 15), name = 'c4_shape')
        self.c4_scale = tf.Variable(tf.random.normal([1]), name = 'c4_scale')
        self.c5_shape = tf.Variable(tf.random.normal([1], mean = 5), name = 'c5_shape')
        self.c5_scale = tf.Variable(tf.random.normal([1]), name = 'c5_scale')
        self.p_shape = tf.Variable(tf.random.normal([1], mean = 5), name = 'p_shape')
        self.p_scale = tf.Variable(tf.random.normal([1]), name = 'p_scale')
        self.ke_shape = tf.Variable(tf.random.normal([1], mean = 10), name = 'ke_shape')
        self.ke_scale = tf.Variable(tf.random.normal([1]), name = 'ke_scale')
        self.ki_shape = tf.Variable(tf.random.normal([1], mean = 10), name = 'ki_shape')
        self.ki_scale = tf.Variable(tf.random.normal([1]), name = 'ki_scale')

    @property
    def qte(self):
        """Variational posterior for TauE"""
        return tfd.Normal(tf.nn.softplus(self.te_shape), tf.nn.softplus(self.te_scale))

    @property
    def qti(self):
        """Variational posterior for TauI"""
        return tfd.Normal(tf.nn.softplus(self.ti_shape), tf.nn.softplus(self.ti_scale))

    @property
    def qc1(self):
        """Variational posterior for c1"""
        return tfd.Normal(tf.nn.softplus(self.c1_shape), tf.nn.softplus(self.c1_scale))

    @property
    def qc2(self):
        """Variational posterior for c2"""
        return tfd.Normal(tf.nn.softplus(self.c2_shape), tf.nn.softplus(self.c2_scale))

    @property
    def qc3(self):
        """Variational posterior for c3"""
        return tfd.Normal(tf.nn.softplus(self.c3_shape), tf.nn.softplus(self.c3_scale))

    @property
    def qc4(self):
        """Variational posterior for c4"""
        return tfd.Normal(tf.nn.softplus(self.c4_shape), tf.nn.softplus(self.c4_scale))

    @property
    def qc5(self):
        """Variational posterior for c5"""
        return tfd.Normal(tf.nn.softplus(self.c5_shape), tf.nn.softplus(self.c5_scale))

    @property
    def qp(self):
        """Variaitonal posterior for P"""
        return tfd.Normal(tf.nn.softplus(self.p_shape), tf.nn.softplus(self.p_scale))
    
    @property
    def qke(self):
        """Variational posterior for kE"""
        return tfd.Normal(tf.nn.softplus(self.ke_shape), tf.nn.softplus(self.ke_scale))

    @property
    def qki(self):
        """Variational posterior for kE"""
        return tfd.Normal(tf.nn.softplus(self.ki_shape), tf.nn.softplus(self.ki_scale))

    @property
    def qstd(self):
        """Variational posterior for sigma"""
        return tfd.InverseGamma(tf.exp(tf.random.normal([1])), tf.exp(tf.random.normal([1])))

    def call(self, obs, sampling = True):
        """Simulate P(Y|Theta) on model call"""
        sample = lambda x: x.sample().numpy() if sampling else x.mean()
        parameters = np.array([
            sample(self.qte),
            sample(self.qti),
            sample(self.qc1),
            sample(self.qc2),
            sample(self.qc3),
            sample(self.qc4),
            sample(self.qc5),
            sample(self.qp),
            sample(self.qke),
            sample(self.qki)
            ])
        _, cohfc = self.NMM.simulate(parameters)
        nmm_coh = cohfc[np.tril_indices(68)].astype('float32')
        # std = tf.sqrt(sample(self.qstd))
        # density = tfd.Normal(nmm_coh, std)
        # return density.log_prob(obs)
        return -pearsonr(obs, nmm_coh)[0].astype('float32')

    @property
    def losses(self):
        """Sum of KL divergences between posterior and priors"""
        prior_ts = tfd.Normal(10,1)
        prior_c1 = tfd.Normal(16,1)
        prior_c2 = tfd.Normal(12,1)
        prior_c3 = tfd.Normal(15,1)
        prior_c4 = tfd.Normal(3,1)
        prior_c5 = tfd.Normal(1.5,1)
        prior_p  = tfd.Normal(1.25, 1)
        prior_ks = tfd.Normal(1,1)
        return (tf.reduce_sum(tfd.kl_divergence(self.qte, prior_ts)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qti, prior_ts)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qc1, prior_c1)) +
                tf.reduce_sum(tfd.kl_divergence(self.qc2, prior_c2)) +
                tf.reduce_sum(tfd.kl_divergence(self.qc3, prior_c3)) +
                tf.reduce_sum(tfd.kl_divergence(self.qc4, prior_c4)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qc5, prior_c5)) +
                tf.reduce_sum(tfd.kl_divergence(self.qp, prior_p)) +
                tf.reduce_sum(tfd.kl_divergence(self.qke, prior_ks)) + 
                tf.reduce_sum(tfd.kl_divergence(self.qki, prior_ks)))