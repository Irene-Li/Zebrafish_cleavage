from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt


class NematicActiveGel2D:
    def __init__(self,
                 width=1,
                 height=1,
                 maxh=0.05,
                 gamma=0.5,
                 eta_1=0.25,
                 eta_2=0,
                 k=0.01,
                 D=0.01,
                 kappa=0.1,
                 beta1=0.1,
                 beta2=0.1,
                 chi0=0.15,
                 chi1=0.1,
                 rho0=1,
                 Qsq=-1):
        """
        Initialize the Active Gel 2D simulation.

        Parameters:
        width, height (float): Domain dimensions
        maxh (float): Maximum mesh size
        gamma (float): Friction coefficient
        eta_1 (float): Incompressible viscosity
        eta_2 (float): Compressible viscosity
        k (float): Turnover rate
        D (float): Diffusion coefficient
        kappa (float): Frank elastic constant
        beta1 (float): Nematic-velocity coupling
        beta2 (float): Nematic relaxation rate
        chi0 (float): Isotropic contractility
        chi1 (float): Anisotropic contractility
        rho0 (float or callable): Steady-state density (may be time-dependent)
        Qsq (float or callable): Spontaneous nematic order (may be time-dependent)
        """
        self.width = width
        self.height = height
        self.maxh = maxh
        self.gamma = gamma
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.k = k
        self.D = D
        self.kappa = kappa
        self.beta1 = beta1
        self.beta2 = beta2
        self.chi0 = chi0
        self.chi1 = chi1
        self.Qsq = Qsq
        self.rho0 = rho0

        self._create_mesh()
        self._setup_function_spaces()
        self._setup_initial_conditions()

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    def _coeff(self, param):
        """Return param(self.time) if callable, else param as-is."""
        return param(self.time) if callable(param) else param

    # -----------------------------------------------------------------------
    # Dimension-aware building blocks
    # Each block returns an NGSolve form expression ready to be accumulated.
    # -----------------------------------------------------------------------

    def _force_balance_bilinear(self, v_trial, v_test):
        if self.mesh.dim == 1:
            return (self.gamma * v_trial * v_test * dx +
                    (self.eta_1 + self.eta_2) * grad(v_trial) * grad(v_test) * dx)
        return (self.gamma * InnerProduct(v_trial, v_test) * dx +
                self.eta_1 * InnerProduct(grad(v_trial), grad(v_test)) * dx +
                self.eta_2 * div(v_trial) * div(v_test) * dx)

    def _density_bilinear(self, rho_trial, rho_test, k=None, D=None):
        if k is None: k = self.k
        if D is None: D = self.D
        if self.mesh.dim == 1:
            return k * rho_trial * rho_test * dx + D * grad(rho_trial) * grad(rho_test) * dx
        return k * rho_trial * rho_test * dx + D * InnerProduct(grad(rho_trial), grad(rho_test)) * dx

    def _nematic_bilinear(self, v_trial, Q_trial, Q_test, q_trial=None, q_test=None):
        Qsq = self._coeff(self.Qsq)
        if self.mesh.dim == 1:
            return ((self.kappa / self.beta2) * grad(Q_trial) * grad(Q_test) * dx
                    + (-Qsq / self.beta2 * Q_trial) * Q_test * dx
                    + self.beta1 / 2 * v_trial * grad(Q_test) * dx)
        Q_time = ((self.kappa / self.beta2) * InnerProduct(grad(Q_trial), grad(Q_test)) * dx
                  + (-Qsq / self.beta2 * Q_trial) * Q_test * dx
                  + self.beta1 / 2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        q_time = ((self.kappa / self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx
                  + (-Qsq / self.beta2 * q_trial) * q_test * dx
                  + self.beta1 / 2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        return Q_time + q_time

    def _advection_nonlinear(self, field_trial, v_trial, field_test):
        """Conservation-form advection: div(field * v) tested against field_test."""
        if self.mesh.dim == 1:
            return (grad(field_trial) * v_trial * field_test * dx +
                    field_trial * grad(v_trial) * field_test * dx)
        return (InnerProduct(grad(field_trial), v_trial) * field_test * dx +
                field_trial * div(v_trial) * field_test * dx)

    def _active_stress_nonlinear(self, v_trial, rho_trial, Q_trial, v_test, q_trial=None):
        """Active stress divergence contribution to the force balance."""
        if self.mesh.dim == 1:
            return (self.chi0 * rho_trial * grad(v_test) * 2 / (rho_trial + 1) * dx
                    + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial * grad(v_test) * dx)
        return (self.chi0 * rho_trial * div(v_test) * 2 / (rho_trial + 1) * dx
                + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (Q_trial * grad(v_test)[0, 0] + q_trial * grad(v_test)[1, 0]) * dx
                + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (-Q_trial * grad(v_test)[1, 1] + q_trial * grad(v_test)[0, 1]) * dx)

    def _nematic_advection_nonlinear(self, v_trial, Q_trial, Q_test, q_trial=None, q_test=None):
        """Co-rotational advection of the Q-tensor."""
        if self.mesh.dim == 1:
            return grad(Q_trial) * v_trial * Q_test * dx
        Q_adv = (InnerProduct(grad(Q_trial), v_trial) * Q_test * dx
                 + (grad(v_trial)[1, 0] - grad(v_trial)[0, 1]) * q_trial * Q_test * dx)
        q_adv = (InnerProduct(grad(q_trial), v_trial) * q_test * dx
                 - (grad(v_trial)[1, 0] - grad(v_trial)[0, 1]) * Q_trial * q_test * dx)
        return Q_adv + q_adv

    def _nematic_hot_nonlinear(self, Q_trial, Q_test, q_trial=None, q_test=None):
        """Higher-order |Q|^2 * Q terms in the nematic equation."""
        if self.mesh.dim == 1:
            return 1 / self.beta2 * Q_test * Q_trial * Q_trial * Q_trial * dx
        Q_hot = 1 / self.beta2 * Q_test * (Q_trial * Q_trial + q_trial * q_trial) * Q_trial * dx
        q_hot = 1 / self.beta2 * q_test * (q_trial * q_trial + Q_trial * Q_trial) * q_trial * dx
        return Q_hot + q_hot

    # -----------------------------------------------------------------------
    # Form assembly (dimension-aware; 1D subclass needs no overrides)
    # -----------------------------------------------------------------------

    def _setup_bilinear_form(self, functions):
        if self.mesh.dim == 1:
            (v_trial, rho_trial, Q_trial), (v_test, rho_test, Q_test) = functions
            return (self._density_bilinear(rho_trial, rho_test)
                    + self._nematic_bilinear(v_trial, Q_trial, Q_test)
                    - self._force_balance_bilinear(v_trial, v_test))
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        return (self._force_balance_bilinear(v_trial, v_test)
                + self._density_bilinear(rho_trial, rho_test)
                + self._nematic_bilinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_nonlinear_form(self, functions):
        if self.mesh.dim == 1:
            (v_trial, rho_trial, Q_trial), (v_test, rho_test, Q_test) = functions
            return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                    + self._nematic_hot_nonlinear(Q_trial, Q_test)
                    + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test)
                    - self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test))
        (v_trial, rho_trial, Q_trial, q_trial), (v_test, rho_test, Q_test, q_test) = functions
        return (self._advection_nonlinear(rho_trial, v_trial, rho_test)
                + self._active_stress_nonlinear(v_trial, rho_trial, Q_trial, v_test, q_trial)
                + self._nematic_hot_nonlinear(Q_trial, Q_test, q_trial, q_test)
                + self._nematic_advection_nonlinear(v_trial, Q_trial, Q_test, q_trial, q_test))

    def _setup_linear_form(self, functions):
        if self.mesh.dim == 1:
            _, (_, rho_test, _) = functions
        else:
            _, (_, rho_test, _, _) = functions
        return self.k * self._coeff(self.rho0) * rho_test * dx

    def _setup_inverse_form(self, functions, bilinear, tau):
        if self.mesh.dim == 1:
            (_, rho_trial, Q_trial), (_, rho_test, Q_test) = functions
            return rho_trial * rho_test * dx + Q_trial * Q_test * dx + tau * bilinear
        (_, rho_trial, Q_trial, q_trial), (_, rho_test, Q_test, q_test) = functions
        return (rho_trial * rho_test * dx + Q_trial * Q_test * dx +
                q_trial * q_test * dx + tau * bilinear)

    # -----------------------------------------------------------------------
    # Mesh, spaces, ICs
    # -----------------------------------------------------------------------

    def _create_mesh(self):
        shape = Rectangle(self.width, self.height).Face()
        shape.edges.Min(X).name = "right"
        shape.edges.Max(X).name = "left"
        shape.edges.Min(Y).name = "up"
        shape.edges.Max(Y).name = "down"
        self.mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=self.maxh)).Curve(3)

    def _setup_function_spaces(self):
        self.V = VectorH1(self.mesh, order=3, dirichlet="right|left|up|down")
        self.R = H1(self.mesh, order=2)
        self.Q = H1(self.mesh, order=2)
        self.q = H1(self.mesh, order=2)
        self.X = self.V * self.R * self.Q * self.q
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.nematic_xx, self.nematic_yx = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        self.density.Set(1)
        self.velocity.Set(CoefficientFunction((0, 0)))
        self.nematic_xx.Set(0)
        self.nematic_yx.Set(0)
        self.time.Set(0)

    # -----------------------------------------------------------------------
    # Public setters
    # -----------------------------------------------------------------------

    def set_initial_density(self, density_function):
        self.density.Set(density_function)

    def set_initial_velocity(self, velocity_function):
        self.velocity.Set(velocity_function)

    # -----------------------------------------------------------------------
    # Simulation
    # -----------------------------------------------------------------------

    def _setup_forms(self, tau):
        functions = self.X.TnT()
        bilinear = self._setup_bilinear_form(functions)
        self.a = BilinearForm(self.X)
        self.a += bilinear
        self.a.Assemble()

        self.nonlinear = BilinearForm(self.X, nonassemble=True)
        self.nonlinear += self._setup_nonlinear_form(functions)

        self.f = LinearForm(self.X)
        self.f += self._setup_linear_form(functions)
        self.f.Assemble()

        mstar = BilinearForm(self.X)
        mstar += self._setup_inverse_form(functions, bilinear, tau)
        mstar.Assemble()
        self.inv = mstar.mat.Inverse(freedofs=self.X.FreeDofs(), inverse="sparsecholesky")

    def simulate(self, tend=10, tau=0.01, save_interval=1):
        t = 0
        self._setup_forms(tau)
        self.gfut = GridFunction(self.gfu.space, multidim=0)
        with TaskManager():
            for i in tqdm(range(int(tend / tau))):
                self.time.Set(t)
                self.nonlinear.Assemble()
                res = (self.a.mat * self.gfu.vec +
                       self.nonlinear.mat * self.gfu.vec -
                       self.f.vec)
                self.gfu.vec.data -= tau * self.inv * res
                if i % save_interval == 0:
                    self.gfut.AddMultiDimComponent(self.gfu.vec)
                t += tau

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def visualize(self, animate=True):
        if animate and hasattr(self, 'gfut'):
            Draw(self.gfut.components[0], self.mesh,
                 interpolate_multidim=True, animate=True, autoscale=True, vectors=True)
            Draw(self.gfut.components[1], self.mesh,
                 interpolate_multidim=True, animate=True, autoscale=True)
            Draw(self.gfut.components[2], self.mesh,
                 interpolate_multidim=True, animate=True, autoscale=True)
        else:
            Draw(self.velocity, self.mesh, "velocity", vectors=True)
            Draw(self.density, self.mesh, "density")
            Draw(self.nematic_xx, self.mesh, "nematic_xx")
            Draw(self.nematic_yx, self.mesh, "nematic_yx")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export(self, filename=None, n_samples=50):
        """Export snapshots to a .npz file with named arrays.

        Returns a dict with keys: rho, vx, vy, Q, q  (each (N, n_samples, n_samples))
        and x, y (1D coordinate arrays).  Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)
        Xc = np.linspace(0, self.width, n_samples)
        Yc = np.linspace(0, self.height, n_samples)
        pts = [(y, x) for x in Xc for y in Yc]

        rho_gf = GridFunction(self.gfut.components[1].space)
        v_gf   = GridFunction(self.gfut.components[0].space)
        Q_gf   = GridFunction(self.gfut.components[2].space)
        q_gf   = GridFunction(self.gfut.components[3].space)

        shape2 = (N, n_samples, n_samples)
        rho_arr = np.zeros(shape2)
        vx_arr  = np.zeros(shape2)
        vy_arr  = np.zeros(shape2)
        Q_arr   = np.zeros(shape2)
        q_arr   = np.zeros(shape2)

        for n in range(N):
            rho_gf.vec.data = self.gfut.components[1].vecs[n]
            v_gf.vec.data   = self.gfut.components[0].vecs[n]
            Q_gf.vec.data   = self.gfut.components[2].vecs[n]
            q_gf.vec.data   = self.gfut.components[3].vecs[n]

            rho_arr[n] = np.array([rho_gf(*p) for p in pts]).reshape(n_samples, n_samples)
            v_vals     = np.array([v_gf(*p)   for p in pts]).reshape(n_samples, n_samples, 2)
            vx_arr[n]  = v_vals[..., 0]
            vy_arr[n]  = v_vals[..., 1]
            Q_arr[n]   = np.array([Q_gf(*p)   for p in pts]).reshape(n_samples, n_samples)
            q_arr[n]   = np.array([q_gf(*p)   for p in pts]).reshape(n_samples, n_samples)

        data = dict(rho=rho_arr, vx=vx_arr, vy=vy_arr, Q=Q_arr, q=q_arr, x=Xc, y=Yc)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, filename, n_samples=50):
        """Deprecated: use export() instead. Kept for backward compatibility."""
        d = self.export(filename, n_samples)
        data = np.stack([d['rho'], d['vx'], d['vy'], d['Q'], d['q']], axis=-1)
        np.save(filename, data)
        return data


# ---------------------------------------------------------------------------

class NematicActiveGel1D(NematicActiveGel2D):
    """1D version. Inherits all dimension-aware form methods from the base class."""

    def _create_mesh(self):
        self.mesh = Make1DMesh(int(1 / self.maxh))

    def _setup_function_spaces(self):
        self.V = H1(self.mesh, order=2, dirichlet="right|left")
        self.R = H1(self.mesh, order=2)
        self.Q = H1(self.mesh, order=2)
        self.X = self.V * self.R * self.Q
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.nematic_xx = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        self.density.Set(1)
        self.velocity.Set(0)
        self.nematic_xx.Set(0)
        self.time.Set(0)

    def export(self, filename=None, n_samples=50):
        """Export snapshots to a .npz file with named arrays.

        Returns a dict with keys: rho, v, Q  (each (N, n_samples)) and x.
        Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)
        Xc = np.linspace(0, 1, n_samples)

        rho_gf = GridFunction(self.R)
        v_gf   = GridFunction(self.V)
        Q_gf   = GridFunction(self.Q)

        rho_arr = np.zeros((N, n_samples))
        v_arr   = np.zeros((N, n_samples))
        Q_arr   = np.zeros((N, n_samples))

        for n in range(N):
            rho_gf.vec.data = self.gfut.components[1].vecs[n]
            v_gf.vec.data   = self.gfut.components[0].vecs[n]
            Q_gf.vec.data   = self.gfut.components[2].vecs[n]
            rho_arr[n] = np.array([rho_gf(x) for x in Xc])
            v_arr[n]   = np.array([v_gf(x)   for x in Xc])
            Q_arr[n]   = np.array([Q_gf(x)   for x in Xc])

        data = dict(rho=rho_arr, v=v_arr, Q=Q_arr, x=Xc)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, n_samples):
        """Deprecated: use export() instead. Kept for backward compatibility."""
        d = self.export(None, n_samples)
        return np.stack([d['rho'], d['v'], d['Q']], axis=-1)


# ---------------------------------------------------------------------------

class ActiveGelCircle(NematicActiveGel2D):

    def _create_mesh(self):
        shape = Circle((0, 0), r=self.width).Face()
        shape.edges.name = 'cyl'
        self.mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=self.maxh)).Curve(3)

    def _setup_function_spaces(self):
        self.V = VectorH1(self.mesh, order=2, dirichlet="cyl")
        self.R = H1(self.mesh, order=2)
        self.Q = H1(self.mesh, order=2)
        self.q = H1(self.mesh, order=2)
        self.X = self.V * self.R * self.Q * self.q
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.nematic_xx, self.nematic_yx = self.gfu.components
        self.time = Parameter(0)

    def export(self, filename=None, n_samples=None):
        """Export snapshots at mesh vertices to a .npz file with named arrays.

        Returns a dict with keys: rho, vx, vy, Q, q  (each (N, n_vertices))
        and mesh_points (n_vertices, 2).  Saves to <filename>.npz when filename is given.
        """
        if not hasattr(self, 'gfut'):
            raise ValueError("No simulation data to export. Run simulate() first.")
        N = len(self.gfut.vecs)

        mesh_points = np.array([[v.point[0], v.point[1]] for v in self.mesh.vertices])
        n_pts = len(mesh_points)

        rho_gf = GridFunction(self.R)
        v_gf   = GridFunction(self.V)
        Q_gf   = GridFunction(self.Q)
        q_gf   = GridFunction(self.q)

        rho_arr = np.zeros((N, n_pts))
        vx_arr  = np.zeros((N, n_pts))
        vy_arr  = np.zeros((N, n_pts))
        Q_arr   = np.zeros((N, n_pts))
        q_arr   = np.zeros((N, n_pts))

        for n in range(N):
            rho_gf.vec.data = self.gfut.components[1].vecs[n]
            v_gf.vec.data   = self.gfut.components[0].vecs[n]
            Q_gf.vec.data   = self.gfut.components[2].vecs[n]
            q_gf.vec.data   = self.gfut.components[3].vecs[n]

            rho_arr[n] = np.array([rho_gf(*p) for p in mesh_points])
            v_vals     = np.array([v_gf(*p)   for p in mesh_points])
            vx_arr[n]  = v_vals[:, 0]
            vy_arr[n]  = v_vals[:, 1]
            Q_arr[n]   = np.array([Q_gf(*p)   for p in mesh_points])
            q_arr[n]   = np.array([q_gf(*p)   for p in mesh_points])

        data = dict(rho=rho_arr, vx=vx_arr, vy=vy_arr, Q=Q_arr, q=q_arr,
                    mesh_points=mesh_points)
        if filename is not None:
            np.savez(filename, **data)
        return data

    def export_to_npy(self, label, n_samples=None):
        """Deprecated: use export() instead. Kept for backward compatibility."""
        d = self.export(label, n_samples)
        data = np.stack([d['rho'], d['vx'], d['vy'], d['Q'], d['q']], axis=-1)
        np.save(label + '_data.npy', data)
        np.save(label + '_mesh.npy', d['mesh_points'])
        return data, d['mesh_points']
