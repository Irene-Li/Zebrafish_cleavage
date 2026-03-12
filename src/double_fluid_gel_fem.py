from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D, NematicActiveGel1D, ActiveGelCircle


class DoubleActinGel2D(NematicActiveGel2D):
    def __init__(self, k_actin=1, rho0_actin=1, **kwargs):
        super().__init__(**kwargs)
        self.k_actin = k_actin
        self.rho0_actin = rho0_actin

    def _setup_function_spaces(self):
        self.V = VectorH1(self.mesh, order=2, dirichlet="right|left|up|down")
        self.R  = H1(self.mesh, order=2)
        self.R2 = H1(self.mesh, order=2)
        self.Q  = H1(self.mesh, order=2)
        self.q  = H1(self.mesh, order=2)
        self.X  = self.V * self.R * self.R2 * self.Q * self.q
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.density2, self.nematic_xx, self.nematic_yx = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        super()._setup_initial_conditions()
        self.density2.Set(1)

    # --- forms (dimension-aware so DoubleFluidGel1D needs no overrides) ---

    def _setup_bilinear_form(self, functions):
        if self.mesh.dim == 1:
            (v_trial, rho_trial, rho2_trial, Q_trial), (v_test, rho_test, rho2_test, Q_test) = functions
            return (self._density_bilinear(rho_trial, rho_test)
                    + self._density_bilinear(rho2_trial, rho2_test, k=self.k_actin)
                    + self._nematic_bilinear(v_trial, Q_trial, Q_test)
                    - self._force_balance_bilinear(v_trial, v_test))
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial), (v_test, rho_test, rho2_test, Q_test, q_test) = functions
        return (self._force_balance_bilinear(v_trial, v_test)
                + self._density_bilinear(rho_trial, rho_test)
                + self._density_bilinear(rho2_trial, rho2_test, k=self.k_actin)
                + self._nematic_bilinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_nonlinear_form(self, functions):
        if self.mesh.dim == 1:
            (v_trial, rho_trial, rho2_trial, Q_trial), (v_test, rho_test, rho2_test, Q_test) = functions
            return (self._advection_nonlinear(rho_trial,  v_trial, rho_test)
                    + self._advection_nonlinear(rho2_trial, v_trial, rho2_test)
                    + self._nematic_hot_nonlinear(Q_trial, Q_test)
                    + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test)
                    - self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test))
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial), (v_test, rho_test, rho2_test, Q_test, q_test) = functions
        return (self._advection_nonlinear(rho_trial,  v_trial, rho_test)
                + self._advection_nonlinear(rho2_trial, v_trial, rho2_test)
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + self._nematic_hot_nonlinear(Q_trial, Q_test, q_trial, q_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_linear_form(self, functions):
        if self.mesh.dim == 1:
            _, (_, rho_test, rho2_test, _) = functions
        else:
            _, (_, rho_test, rho2_test, _, _) = functions
        return (self.k * self._coeff(self.rho0) * rho_test * dx
                + self.k_actin * self.rho0_actin * rho2_test * dx)

    def _setup_inverse_form(self, functions, bilinear, tau):
        if self.mesh.dim == 1:
            (_, rho_trial, rho2_trial, Q_trial), (_, rho_test, rho2_test, Q_test) = functions
            return (rho_trial * rho_test * dx + rho2_trial * rho2_test * dx
                    + Q_trial * Q_test * dx + tau * bilinear)
        (_, rho_trial, rho2_trial, Q_trial, q_trial), (_, rho_test, rho2_test, Q_test, q_test) = functions
        return (rho_trial * rho_test * dx + rho2_trial * rho2_test * dx
                + Q_trial * Q_test * dx + q_trial * q_test * dx + tau * bilinear)

    def visualize(self, animate=True):
        if animate and hasattr(self, 'gfut'):
            Draw(self.gfut.components[0], self.mesh,
                 interpolate_multidim=True, animate=True, autoscale=True, vectors=True)
            for i in range(1, 4):
                Draw(self.gfut.components[i], self.mesh,
                     interpolate_multidim=True, animate=True, autoscale=True)
        else:
            Draw(self.velocity,     self.mesh, "velocity",    vectors=True)
            Draw(self.density,      self.mesh, "density")
            Draw(self.density2,     self.mesh, "density2")
            Draw(self.nematic_xx,   self.mesh, "nematic_xx")
            Draw(self.nematic_yx,   self.mesh, "nematic_yx")

    def export(self, filename=None, n_samples=50):
        """Export snapshots to a .npz file with named arrays.

        Returns a dict with keys: rho, rho2, vx, vy, Q, q  (each (N, n_samples, n_samples))
        and x, y.  Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)
        Xc = np.linspace(0, self.width,  n_samples)
        Yc = np.linspace(0, self.height, n_samples)
        pts = [(y, x) for x in Xc for y in Yc]

        rho_gf  = GridFunction(self.R)
        rho2_gf = GridFunction(self.R2)
        v_gf    = GridFunction(self.V)
        Q_gf    = GridFunction(self.Q)
        q_gf    = GridFunction(self.q)

        shape2 = (N, n_samples, n_samples)
        rho_arr  = np.zeros(shape2)
        rho2_arr = np.zeros(shape2)
        vx_arr   = np.zeros(shape2)
        vy_arr   = np.zeros(shape2)
        Q_arr    = np.zeros(shape2)
        q_arr    = np.zeros(shape2)

        for n in range(N):
            rho_gf.vec.data  = self.gfut.components[1].vecs[n]
            rho2_gf.vec.data = self.gfut.components[2].vecs[n]
            v_gf.vec.data    = self.gfut.components[0].vecs[n]
            Q_gf.vec.data    = self.gfut.components[3].vecs[n]
            q_gf.vec.data    = self.gfut.components[4].vecs[n]

            rho_arr[n]  = np.array([rho_gf(*p)  for p in pts]).reshape(n_samples, n_samples)
            rho2_arr[n] = np.array([rho2_gf(*p) for p in pts]).reshape(n_samples, n_samples)
            v_vals      = np.array([v_gf(*p)    for p in pts]).reshape(n_samples, n_samples, 2)
            vx_arr[n]   = v_vals[..., 0]
            vy_arr[n]   = v_vals[..., 1]
            Q_arr[n]    = np.array([Q_gf(*p)    for p in pts]).reshape(n_samples, n_samples)
            q_arr[n]    = np.array([q_gf(*p)    for p in pts]).reshape(n_samples, n_samples)

        data = dict(rho=rho_arr, rho2=rho2_arr, vx=vx_arr, vy=vy_arr, Q=Q_arr, q=q_arr,
                    x=Xc, y=Yc)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, filename, n_samples=50):
        """Deprecated: use export() instead."""
        d = self.export(filename, n_samples)
        data = np.stack([d['rho'], d['rho2'], d['vx'], d['vy'], d['Q'], d['q']], axis=-1)
        np.save(filename, data)
        return data


# ---------------------------------------------------------------------------

class UnsaturedMonomerModel(DoubleActinGel2D):
    def __init__(self, k2=0.1, k4=0.1, k5=0.1, **kwargs):
        super().__init__(**kwargs)
        self.k2 = k2
        self.k4 = k4
        self.k5 = k5

    def _setup_function_spaces(self):
        self.V  = VectorH1(self.mesh, order=2, dirichlet="right|left|up|down")
        self.R  = H1(self.mesh, order=2)
        self.R2 = H1(self.mesh, order=2)
        self.R3 = H1(self.mesh, order=2)
        self.Q  = H1(self.mesh, order=2)
        self.q  = H1(self.mesh, order=2)
        self.X  = self.V * self.R * self.R2 * self.Q * self.q * self.R3
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.density2, self.nematic_xx, self.nematic_yx, self.m = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        super()._setup_initial_conditions()
        self.m.Set(1)

    def _setup_bilinear_form(self, functions):
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial, m_trial), \
        (v_test,  rho_test,  rho2_test,  Q_test,  q_test,  m_test) = functions

        lhs_rho  = (self._density_bilinear(rho_trial, rho_test)
                    - self.k2 * m_trial * rho_test * dx)
        lhs_rho2 = (self._density_bilinear(rho2_trial, rho2_test, k=self.k_actin)
                    - self.k4 * m_trial * rho2_test * dx)
        lhs_m    = ((- self.k * rho_trial - self.k_actin * rho2_trial
                     + self.k2 * m_trial + self.k4 * m_trial + self.k5 * m_trial)
                    * m_test * dx)

        return (self._force_balance_bilinear(v_trial, v_test)
                + lhs_rho + lhs_rho2
                + self._nematic_bilinear(v_trial, Q_trial, Q_test, q_trial, q_test)
                + lhs_m)

    def _setup_nonlinear_form(self, functions):
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial, m_trial), \
        (v_test,  rho_test,  rho2_test,  Q_test,  q_test,  m_test) = functions

        return (self._advection_nonlinear(rho_trial,  v_trial, rho_test)
                + self._advection_nonlinear(rho2_trial, v_trial, rho2_test)
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + self._nematic_hot_nonlinear(Q_trial, Q_test, q_trial, q_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_linear_form(self, functions):
        _, (_, _, _, _, _, m_test) = functions
        return self.k5 * m_test * dx

    def _setup_inverse_form(self, functions, bilinear, tau):
        (_, rho_trial, rho2_trial, Q_trial, q_trial, m_trial), \
        (_, rho_test,  rho2_test,  Q_test,  q_test,  m_test) = functions
        return (rho_trial * rho_test * dx + rho2_trial * rho2_test * dx
                + m_trial * m_test * dx
                + Q_trial * Q_test * dx + q_trial * q_test * dx
                + tau * bilinear)


# ---------------------------------------------------------------------------

class DoubleFluidGel1D(DoubleActinGel2D):
    """1D version. Inherits dimension-aware form methods from DoubleActinGel2D."""

    def _create_mesh(self):
        self.mesh = Make1DMesh(int(1 / self.maxh))

    def _setup_function_spaces(self):
        self.V  = H1(self.mesh, order=2, dirichlet="right|left")
        self.R  = H1(self.mesh, order=2)
        self.R2 = H1(self.mesh, order=2)
        self.Q  = H1(self.mesh, order=2)
        self.X  = self.V * self.R * self.R2 * self.Q
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.density2, self.nematic_xx = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        self.density.Set(0)
        self.density2.Set(1)
        self.velocity.Set(0)
        self.nematic_xx.Set(0)
        self.time.Set(0)

    def export(self, filename=None, n_samples=50):
        """Export snapshots to a .npz file with named arrays.

        Returns a dict with keys: rho, rho2, v, Q  (each (N, n_samples)) and x.
        Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)
        Xc = np.linspace(0, 1, n_samples)

        rho_gf  = GridFunction(self.R)
        rho2_gf = GridFunction(self.R2)
        v_gf    = GridFunction(self.V)
        Q_gf    = GridFunction(self.Q)

        rho_arr  = np.zeros((N, n_samples))
        rho2_arr = np.zeros((N, n_samples))
        v_arr    = np.zeros((N, n_samples))
        Q_arr    = np.zeros((N, n_samples))

        for n in range(N):
            rho_gf.vec.data  = self.gfut.components[1].vecs[n]
            rho2_gf.vec.data = self.gfut.components[2].vecs[n]
            v_gf.vec.data    = self.gfut.components[0].vecs[n]
            Q_gf.vec.data    = self.gfut.components[3].vecs[n]
            rho_arr[n]  = np.array([rho_gf(x)  for x in Xc])
            rho2_arr[n] = np.array([rho2_gf(x) for x in Xc])
            v_arr[n]    = np.array([v_gf(x)    for x in Xc])
            Q_arr[n]    = np.array([Q_gf(x)    for x in Xc])

        data = dict(rho=rho_arr, rho2=rho2_arr, v=v_arr, Q=Q_arr, x=Xc)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, n_samples):
        """Deprecated: use export() instead."""
        d = self.export(None, n_samples)
        return np.stack([d['rho'], d['rho2'], d['v'], d['Q']], axis=-1)
