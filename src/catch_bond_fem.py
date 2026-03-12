from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D, NematicActiveGel1D, ActiveGelCircle

tanh = lambda arg: (exp(arg) - exp(-arg)) / (exp(arg) + exp(-arg))


class CatchBond1D(NematicActiveGel1D):
    """1D catch-bond model.

    Difference from base: nematic bilinear uses +1/beta2 (not -Qsq/beta2),
    and the HOT term is replaced by a tanh-based catch-bond alignment.
    """

    def _setup_bilinear_form(self, functions):
        (v_trial, rho_trial, Q_trial), (v_test, rho_test, Q_test) = functions
        Q_time = ((self.kappa / self.beta2) * grad(Q_trial) * grad(Q_test) * dx
                  + 1 / self.beta2 * Q_trial * Q_test * dx
                  + self.beta1 / 2 * v_trial * grad(Q_test) * dx)
        return (self._density_bilinear(rho_trial, rho_test)
                + Q_time
                - self._force_balance_bilinear(v_trial, v_test))

    def _setup_nonlinear_form(self, functions):
        (v_trial, rho_trial, Q_trial), (v_test, rho_test, Q_test) = functions
        T = self.eta_1 * grad(v_trial) / 2 + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial
        Q_cb = -1 / self.beta2 * tanh(T / 0.1) * Q_test * self._coeff(self.Qsq) * dx
        return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test)
                + Q_cb
                - self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test))


class CatchBond2D(NematicActiveGel2D):
    """2D catch-bond model.

    Difference from base: nematic bilinear uses +1/beta2 (not -Qsq/beta2),
    and the HOT term is replaced by tanh-based catch-bond alignment.
    """

    def _setup_bilinear_form(self, functions):
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        Q_time = ((self.kappa / self.beta2) * InnerProduct(grad(Q_trial), grad(Q_test)) * dx
                  + (1 / self.beta2 * Q_trial) * Q_test * dx
                  + self.beta1 / 2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        q_time = ((self.kappa / self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx
                  + (1 / self.beta2 * q_trial) * q_test * dx
                  + self.beta1 / 2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        return (self._force_balance_bilinear(v_trial, v_test)
                + self._density_bilinear(rho_trial, rho_test)
                + Q_time + q_time)

    def _setup_nonlinear_form(self, functions):
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        Txx = (self.eta_1 * (grad(v_trial)[0, 0] - grad(v_trial)[1, 1]) / 2
               + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial)
        Txy = (self.eta_1 * (grad(v_trial)[0, 1] + grad(v_trial)[1, 0]) / 2
               + self.chi1 * rho_trial * 2 / (rho_trial + 1) * q_trial)
        Qsq = self._coeff(self.Qsq)
        Q_hot = -1 / self.beta2 * Q_test * tanh(Txx / 0.1) * Qsq * dx
        q_hot = -1 / self.beta2 * q_test * tanh(Txy / 0.1) * Qsq * dx
        return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + Q_hot + q_hot
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))
