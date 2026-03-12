from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D, NematicActiveGel1D


class MyosinActinGel2D(NematicActiveGel2D):
    def __init__(self,
                 S=1,
                 m0=1,
                 k_m=0.01,
                 D_rho=0.01,
                 D_m=0.01,
                 k0=0.01,
                 k1=0.01,
                 **kwargs):
        """
        Coupled actin-myosin gel model:
            d_t rho + div(v*rho) = S(x) - (k0 + k1*m)*rho + D_rho * laplacian(rho)
            d_t m   + div(v*m)   = -k_m*(m - m0(x))       + D_m   * laplacian(m)

        Parameters (in addition to base class):
        S    : Source term (NGSolve CF or scalar)
        m0   : Myosin target field (NGSolve CF or scalar)
        k_m  : Myosin relaxation rate
        D_rho: Density diffusion coefficient
        D_m  : Myosin diffusion coefficient
        k0   : Baseline absorption rate  (stored as k0_abs to avoid clash with base k)
        k1   : Myosin-dependent absorption rate
        """
        self.S      = S
        self.m0     = m0
        self.k_m    = k_m
        self.D_rho  = D_rho
        self.D_m    = D_m
        self.k0_abs = k0
        self.k1     = k1
        super().__init__(**kwargs)

    def _setup_function_spaces(self):
        self.V       = VectorH1(self.mesh, order=3, dirichlet="right|left|up|down")
        self.R       = H1(self.mesh, order=2)
        self.M_space = H1(self.mesh, order=2)
        self.Q       = H1(self.mesh, order=2)
        self.q       = H1(self.mesh, order=2)
        self.X       = self.V * self.R * self.M_space * self.Q * self.q
        self.gfu     = GridFunction(self.X)
        self.velocity, self.density, self.myosin, self.nematic_xx, self.nematic_yx = self.gfu.components
        self.time    = Parameter(0)

    def _setup_initial_conditions(self):
        self.density.Set(1)
        self.myosin.Set(self.m0)
        self.velocity.Set(CoefficientFunction((0, 0)))
        self.nematic_xx.Set(0)
        self.nematic_yx.Set(0)
        self.time.Set(0)

    # --- forms (dimension-aware so MyosinActinGel1D needs no overrides) ---

    def _setup_bilinear_form(self, functions):
        if self.mesh.dim == 1:
            (v_trial, rho_trial, m_trial, Q_trial), (v_test, rho_test, m_test, Q_test) = functions
            return (self._density_bilinear(rho_trial, rho_test, k=self.k0_abs, D=self.D_rho)
                    + self._density_bilinear(m_trial,   m_test,   k=self.k_m,   D=self.D_m)
                    + self._nematic_bilinear(v_trial, Q_trial, Q_test)
                    - self._force_balance_bilinear(v_trial, v_test))
        (v_trial, rho_trial, m_trial, Q_trial, q_trial), (v_test, rho_test, m_test, Q_test, q_test) = functions
        return (self._force_balance_bilinear(v_trial, v_test)
                + self._density_bilinear(rho_trial, rho_test, k=self.k0_abs, D=self.D_rho)
                + self._density_bilinear(m_trial,   m_test,   k=self.k_m,   D=self.D_m)
                + self._nematic_bilinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_nonlinear_form(self, functions):
        if self.mesh.dim == 1:
            (v_trial, rho_trial, m_trial, Q_trial), (v_test, rho_test, m_test, Q_test) = functions
            coupling = self.k1 * m_trial * rho_trial * rho_test * dx
            return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                    + coupling
                    + self._advection_nonlinear(m_trial, v_trial, m_test)
                    + self._nematic_hot_nonlinear(Q_trial, Q_test)
                    + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test)
                    - self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test))
        (v_trial, rho_trial, m_trial, Q_trial, q_trial), (v_test, rho_test, m_test, Q_test, q_test) = functions
        coupling = self.k1 * m_trial * rho_trial * rho_test * dx
        return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                + coupling
                + self._advection_nonlinear(m_trial, v_trial, m_test)
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + self._nematic_hot_nonlinear(Q_trial, Q_test, q_trial, q_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_linear_form(self, functions):
        if self.mesh.dim == 1:
            _, (_, rho_test, m_test, _) = functions
        else:
            _, (_, rho_test, m_test, _, _) = functions
        return self.S * rho_test * dx + self.k_m * self.m0 * m_test * dx

    def _setup_inverse_form(self, functions, bilinear, tau):
        if self.mesh.dim == 1:
            (_, rho_trial, m_trial, Q_trial), (_, rho_test, m_test, Q_test) = functions
            return (rho_trial * rho_test * dx + m_trial * m_test * dx
                    + Q_trial * Q_test * dx + tau * bilinear)
        (_, rho_trial, m_trial, Q_trial, q_trial), (_, rho_test, m_test, Q_test, q_test) = functions
        return (rho_trial * rho_test * dx + m_trial * m_test * dx
                + Q_trial * Q_test * dx + q_trial * q_test * dx + tau * bilinear)

    # --- setters ---

    def set_source(self, S_func):
        self.S = S_func

    def set_m0(self, m0_func):
        self.m0 = m0_func

    def set_initial_myosin(self, m_func):
        self.myosin.Set(m_func)

    # --- visualise ---

    def visualize(self, animate=True):
        if animate and hasattr(self, 'gfut'):
            Draw(self.gfut.components[0], self.mesh,
                 interpolate_multidim=True, animate=True, autoscale=True, vectors=True)
            for i in range(1, 5):
                Draw(self.gfut.components[i], self.mesh,
                     interpolate_multidim=True, animate=True, autoscale=True)
        else:
            Draw(self.velocity,    self.mesh, "velocity",    vectors=True)
            Draw(self.density,     self.mesh, "density")
            Draw(self.myosin,      self.mesh, "myosin")
            Draw(self.nematic_xx,  self.mesh, "nematic_xx")
            Draw(self.nematic_yx,  self.mesh, "nematic_yx")

    # --- export ---

    def export(self, filename=None, n_samples=50):
        """Export snapshots to a .npz file with named arrays.

        Returns a dict with keys: rho, m, vx, vy, Q, q  (each (N, n_samples, n_samples))
        and x, y.  Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)
        Xc = np.linspace(0, self.width,  n_samples)
        Yc = np.linspace(0, self.height, n_samples)
        pts = [(y, x) for x in Xc for y in Yc]

        rho_gf = GridFunction(self.R)
        m_gf   = GridFunction(self.M_space)
        v_gf   = GridFunction(self.V)
        Q_gf   = GridFunction(self.Q)
        q_gf   = GridFunction(self.q)

        shape2 = (N, n_samples, n_samples)
        rho_arr = np.zeros(shape2)
        m_arr   = np.zeros(shape2)
        vx_arr  = np.zeros(shape2)
        vy_arr  = np.zeros(shape2)
        Q_arr   = np.zeros(shape2)
        q_arr   = np.zeros(shape2)

        for n in range(N):
            rho_gf.vec.data = self.gfut.components[1].vecs[n]
            m_gf.vec.data   = self.gfut.components[2].vecs[n]
            v_gf.vec.data   = self.gfut.components[0].vecs[n]
            Q_gf.vec.data   = self.gfut.components[3].vecs[n]
            q_gf.vec.data   = self.gfut.components[4].vecs[n]

            rho_arr[n] = np.array([rho_gf(*p) for p in pts]).reshape(n_samples, n_samples)
            m_arr[n]   = np.array([m_gf(*p)   for p in pts]).reshape(n_samples, n_samples)
            v_vals     = np.array([v_gf(*p)   for p in pts]).reshape(n_samples, n_samples, 2)
            vx_arr[n]  = v_vals[..., 0]
            vy_arr[n]  = v_vals[..., 1]
            Q_arr[n]   = np.array([Q_gf(*p)   for p in pts]).reshape(n_samples, n_samples)
            q_arr[n]   = np.array([q_gf(*p)   for p in pts]).reshape(n_samples, n_samples)

        data = dict(rho=rho_arr, m=m_arr, vx=vx_arr, vy=vy_arr, Q=Q_arr, q=q_arr,
                    x=Xc, y=Yc)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, filename, n_samples=50):
        """Deprecated: use export() instead."""
        d = self.export(filename, n_samples)
        data = np.stack([d['rho'], d['m'], d['vx'], d['vy'], d['Q'], d['q']], axis=-1)
        np.save(filename, data)
        return data


# ---------------------------------------------------------------------------

class MyosinActinGel1D(MyosinActinGel2D):
    """1D version. Inherits dimension-aware form methods from MyosinActinGel2D."""

    def _create_mesh(self):
        self.mesh = Make1DMesh(int(1 / self.maxh))

    def _setup_function_spaces(self):
        self.V       = H1(self.mesh, order=2, dirichlet="right|left")
        self.R       = H1(self.mesh, order=2)
        self.M_space = H1(self.mesh, order=2)
        self.Q       = H1(self.mesh, order=2)
        self.X       = self.V * self.R * self.M_space * self.Q
        self.gfu     = GridFunction(self.X)
        self.velocity, self.density, self.myosin, self.nematic_xx = self.gfu.components
        self.time    = Parameter(0)

    def _setup_initial_conditions(self):
        self.density.Set(1)
        self.myosin.Set(self.m0)
        self.velocity.Set(0)
        self.nematic_xx.Set(0)
        self.time.Set(0)

    def export(self, filename=None, n_samples=50):
        """Export snapshots to a .npz file with named arrays.

        Returns a dict with keys: rho, m, v, Q  (each (N, n_samples)) and x.
        Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)
        Xc = np.linspace(0, 1, n_samples)

        rho_gf = GridFunction(self.R)
        m_gf   = GridFunction(self.M_space)
        v_gf   = GridFunction(self.V)
        Q_gf   = GridFunction(self.Q)

        rho_arr = np.zeros((N, n_samples))
        m_arr   = np.zeros((N, n_samples))
        v_arr   = np.zeros((N, n_samples))
        Q_arr   = np.zeros((N, n_samples))

        for n in range(N):
            rho_gf.vec.data = self.gfut.components[1].vecs[n]
            m_gf.vec.data   = self.gfut.components[2].vecs[n]
            v_gf.vec.data   = self.gfut.components[0].vecs[n]
            Q_gf.vec.data   = self.gfut.components[3].vecs[n]
            rho_arr[n] = np.array([rho_gf(x) for x in Xc])
            m_arr[n]   = np.array([m_gf(x)   for x in Xc])
            v_arr[n]   = np.array([v_gf(x)   for x in Xc])
            Q_arr[n]   = np.array([Q_gf(x)   for x in Xc])

        data = dict(rho=rho_arr, m=m_arr, v=v_arr, Q=Q_arr, x=Xc)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, n_samples):
        """Deprecated: use export() instead."""
        d = self.export(None, n_samples)
        return np.stack([d['rho'], d['m'], d['v'], d['Q']], axis=-1)
