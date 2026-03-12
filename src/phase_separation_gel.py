from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D, NematicActiveGel1D


class NonconsPhaseSeparation2D(NematicActiveGel2D):
    """Non-conserved phase separation: adds a cubic rho nonlinearity."""

    def _setup_bilinear_form(self, functions):
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        lhs_rho = self._density_bilinear(rho_trial, rho_test,
                                         k=self.k * 0.5 * self._coeff(self.rho0))
        return (self._force_balance_bilinear(v_trial, v_test)
                + lhs_rho
                + self._nematic_bilinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_nonlinear_form(self, functions):
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        rho_hot = self.k * rho_test * (rho_trial * rho_trial * rho_trial
                                       - (self._coeff(self.rho0) + 0.5) * rho_trial) * dx
        return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                + rho_hot
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + self._nematic_hot_nonlinear(Q_trial, Q_test, q_trial, q_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_linear_form(self, functions):
        _, (_, rho_test, _, _) = functions
        return 0 * rho_test * dx


class ConsPhaseSeparation2D(NematicActiveGel2D):
    """Conserved phase separation: adds a biharmonic rho regularisation."""

    def _setup_bilinear_form(self, functions):
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        aux_trial = rho_trial.Operator('hesse')
        aux_test  = rho_test.Operator('hesse')
        lhs_rho = (self.k * rho_trial * rho_test * dx
                   + self.D * (0.5 * InnerProduct(grad(rho_trial), grad(rho_test))
                                + self.kappa * (aux_trial[0, 0] + aux_trial[1, 1])
                                             * (aux_test[0, 0]  + aux_test[1, 1])) * dx)
        return (self._force_balance_bilinear(v_trial, v_test)
                + lhs_rho
                + self._nematic_bilinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_nonlinear_form(self, functions):
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        rho_hot = self.D * InnerProduct(grad(rho_test),
                                        grad(rho_trial * rho_trial * rho_trial)
                                        - 1.5 * grad(rho_trial * rho_trial)) * dx
        return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                + rho_hot
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + self._nematic_hot_nonlinear(Q_trial, Q_test, q_trial, q_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_linear_form(self, functions):
        _, (_, rho_test, _, _) = functions
        return self.k * self._coeff(self.rho0) * rho_test * dx
