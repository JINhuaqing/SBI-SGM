from jitcdde import t, y
from matplotlib.pyplot import yscale
from symengine import exp, var
import functions.utils as utils
import numpy as np
import time

class NetworkModel:
    def __init__(self, times, Cjk, Djk, fmin=2, fmax=45):
        self._times = times
        self._C = Cjk
        self._D = Djk 
        self._delays = Djk/10 # Assuming 10meters/sec uniform transmission speed
        self._max_delays = np.round(np.amax(self._delays), 3)
        self._initial_state = np.zeros(len(Cjk)*2)
        self._initial_deriv = np.zeros(len(Cjk)*2)
        self._ae = 1.3
        self._ai = 2
        self._thE = 4
        self._thI = 3.7
        self.num_nodes = 68

    def Model(self):
        var("tauE tauI c1 c2 c3 c4 c5 kE kI")
        for j in range(68):
            coupling_sum = c5 * sum(
                self._C[j,k] * y(2 * k, t - self._delays[j,k]) for k in range(68) if self._C[j,k]
            )
            # sigmoids:
            xE = c1 * y(2 * j) - c2 * y(2 * j + 1) + coupling_sum
            xI = c3 * y(2 * j) - c4 * y(2 * j + 1)
            Se = (1 / (1 + exp(-self._ae * (xE - self._thE)))) - (1 / (1 + exp(self._ae * self._thE)))
            Si = (1 / (1 + exp(-self._ai * (xI - self._thI)))) - (1 / (1 + exp(self._ai * self._thI)))
            # model derivatives
            yield (-y(2 * j) + (kE - y(2 * j)) * Se)/tauE
            yield (-y(2 * j + 1) + (kI - y(2 * j + 1)) * Si)/tauI

    def nmm_ioc(self):
        var("tau_e tau_i c1 c2 c3 c4 c5 pt")
        for j in range(self.num_nodes):
            coupling_sum = c5 * sum(
                self._C[j, k] * y(2 * k, t - self._delays[j, k]) for k in range(self.num_nodes) if
                self._C[j, k])
            x_e = c1 * y(2 * j) - c2 * y(2 * j + 1) + coupling_sum + pt
            x_i = c3 * y(2 * j) - c4 * y(2 * j + 1)
            # Sigmoid Function:
            s_e = (1 / (1 + exp(-self._ae * (x_e - self._thE)))) - (1 / (1 + exp(self._ae * self._thE)))
            s_i = (1 / (1 + exp(-self._ai * (x_i - self._thI)))) - (1 / (1 + exp(self._ai * self._thI)))
            # derivatives:
            # simple model
            yield (-y(2 * j) + (1 - y(2 * j)) * s_e) / tau_e
            yield (-y(2 * j + 1) + (1 - y(2 * j + 1)) * s_i) / tau_i
    # def compile(self):
    #     self.DDE = jitcdde(self._WilsonCowanNetwork, n = 68*2, max_delay = 14.71, control_pars=var("tauE tauI c1 c2 c3 c4 c5 kE kI"))
    #     params = (8,8,16,12,15,3,1.5,1,1)
    #     start_time = time.time()
    #     self.DDE.add_past_point(-0.01, self._initial_state, self._initial_deriv)
    #     self.DDE.add_past_point(0, self._initial_state, self._initial_deriv)
    #     self.DDE.set_parameters(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8])
    #     self.DDE.set_integration_parameters(rtol=1e-6, atol=1e-6, max_step=10)
    #     self.DDE.integrate_blindly(0.02, 0.01)
    #     print("Compile integrator time: {:.3f}".format(time.time()-start_time))
