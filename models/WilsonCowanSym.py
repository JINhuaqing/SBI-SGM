import numpy as np 
import functions.utils as utils

from spectrum import dpss
from jitcdde import t, jitcdde
from jitcdde import y as ddestate
from jitcode import y, jitcode
from symengine import exp, var, Function, Symbol
from matplotlib.pyplot import subplots, show

np.random.seed(111)

class WilsonCowanOscillator:
    def __init__(self, times, nfft, fmin=2, fmax=45):
        self._times = times
        # self._y0 = np.random.randn(2)
        self._y0 = np.array([0.1, 0.1]) 
        self._dy = np.zeros(2)
        self._nfft = nfft
        self._fs = int(np.round(1/(times[1] - times[0])))
        self.findices, self.frequencies = utils.get_frequencies(nfft, self._fs, fmin=fmin, fmax=fmax)
        self.tapers, self.eigen = dpss(len(times), 3)

    def compile(self):
        """Compile C code with jitcode symengine equations"""
        # tauE, tauI, c1, c2, c3, c4, kE, kI, P = [float(x) for x in parameters]
        rE = 1
        rI = 1
        aE = 1.3
        aI = 2
        thrE = 4
        thrI = 3.7
        Q  = 0
        noise = Function("noise")
        # var("tauE tauI c1 c2 c3 c4 kE kI P")

        def make_noise(y, scale):
            return scale * np.random.randn()

        def WilsonCowan():
            E, I = y(0), y(1)
            # sigmoids:
            xE = c1 * E - c2 * I + P
            xI = c3 * E - c4 * I + Q
            Se = 1 / (1 + exp(-aE * (xE - thrE))) - 1 / (1 + exp(aE*thrE))
            Si = 1 / (1 + exp(-aI * (xI - thrI))) - 1 / (1 + exp(aI*thrI)) 
            # model derivatives
            yield (-E + (kE - rE * E) * Se + noise(0.00001))/tauE
            yield (-I + (kI - rI * I) * Si + noise(0.00001))/tauI

        self.ODE = jitcode(WilsonCowan, n = 2, control_pars=var("tauE tauI c1 c2 c3 c4 kE kI P"), callback_functions=[(noise, make_noise, 1)])
        self.ODE.set_integrator("dopri5")
        self.ODE.set_initial_value([0.1, 0.1])
        # , callback_functions=[(noise, make_noise, 0)]
        # self.ODE.set_integration_parameters(rtol=1e-6, atol=1e-6, max_step=10)
        # self.ODE.check()
        # self.ODE.constant_past([0.1, 0.1])
        # self.ODE.step_on_discontinuities()
        # self.ODE.integrate_blindly(0.02, 0.01)

    def _simulate(self, parameters):
        self.ODE.set_parameters(parameters)
        mod_out = [self.ODE.integrate(time) for time in self.ODE.t + self._times]
        # self.ODE.purge_past()
        return np.asarray(mod_out)

    def _compute_pmtm(self, timecourse):
        return utils.get_multitaper(timecourse, self._nfft, self.eigen, self.tapers)

    def simulate(self, parameters):
        ymodel = self._simulate(parameters)
        ymodel = ymodel[:,0] - ymodel[:,1]
        full_spectrum = self._compute_pmtm(ymodel)[self.findices]
        self.model_timecourse = ymodel 
        self.model_spectrum = full_spectrum
        return ymodel, full_spectrum

    def visualize(self):
        f, a = subplots(1,2, figsize=(10,2.5), constrained_layout=True)
        a[0].plot(self._times/1000, self.model_timecourse)
        a[0].set_title('Timecourse')
        a[0].set_xlabel('time (s)')
        a[1].plot(self.frequencies, np.log10(self.model_spectrum))
        a[1].set_title('Multitaper Spectrum')
        a[1].set_xlabel('Frequencies (Hz)')
        a[1].set_ylabel('Log Scale Power')
        show()


class WilsonCowanNetwork:
    def __init__(self, times, Cjk, Djk, fmin=2, fmax=45):
        self._times = times
        self._C = Cjk
        self._D = Djk
        self._delays = Djk/10 # Assuming 10meters/sec uniform transmission speed
        self._max_delays = np.round(np.amax(self._delays), 3)
        self._initial_state = np.zeros(len(Cjk)*2)
        self._initial_deriv = np.zeros(len(Cjk)*2)

    def compile(self):
        """Compile c code with jitcdde"""
        rE = 1
        rI = 1
        aE = 1.3
        aI = 2
        thrE = 4
        thrI = 3.7
        P = 1.25
        Q = 0

        noise = Function("noise")

        def make_noise(y, scale):
            return scale * np.random.randn()

        def WilsonCowan():
            var("tau_e tau_i c1 c2 c3 c4 c5 kE kI")
            for j in range(68):
                coupling_sum = c5 * sum(self._C[j,k] * ddestate(2*k, t - self._delays[j,k]) for k in range(68) if self._C[j,k])
                # sigmoids:
                xE = c1 * ddestate(2*j) - c2 * ddestate(2*j+1) + coupling_sum + P
                xI = c3 * ddestate(2*j) - c4 * ddestate(2*j+1)
                Se = (1 / (1 + exp(-aE * (xE - thrE)))) - (1 / (1 + exp(aE*thrE)))
                Si = (1 / (1 + exp(-aI * (xI - thrI)))) - (1 / (1 + exp(aI*thrI)))
                # model derivatives
                yield (-ddestate(2*j) + (kE - rE * ddestate(2*j)) * Se + noise(0.00001))/tauE
                yield (-ddestate(2*j+1) + (kI - rI * ddestate(2*j+1)) * Si + noise(0.00001))/tauI

        self.DDE = jitcdde(WilsonCowan, n = 68*2, max_delay = 14.71, control_pars=var("tauE tauI c1 c2 c3 c4 c5 kE kI"), callback_functions=[(noise, make_noise, 1)])
        self.DDE.set_integration_parameters(rtol=0, atol=1e-6, max_step=10)

    def _simulate(self, parameters):
        self.DDE.constant_past(self._initial_state)
        self.DDE.set_parameters(parameters)
        self.DDE.integrate_blindly(0.02, 0.01)
        mod_out = [self.DDE.integrate(time) for time in self.DDE.t + self._times]
        self.DDE.purge_past()
        return np.asarray(mod_out)