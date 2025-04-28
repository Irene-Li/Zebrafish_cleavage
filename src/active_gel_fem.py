from ngsolve import *
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
                 alpha = 0.1,
                 kappa = 0.1,
                 beta1 = 0.1,
                 beta2 = 0.1,
                 chi0 = 0.15, 
                 chi1 = 0.1,
                 Qsq = - 1):
        """
        Initialize the Active Gel 2D simulation.
        
        Parameters:
        width (float): Width of the rectangular domain
        height (float): Height of the rectangular domain
        maxh (float): Maximum mesh size
        gamma (float): Friction coefficient
        eta_1 (float): imcompressible viscosity coefficient
        eta_2 (float): compressible viscosity coefficient
        k (float): turnover rate 
        D (float): Diffusion coefficient
        alpha: restoration rate of nematic order 
        kappa: alignment rate of nematic order 
        beta1 (float): alignment of nematic order with velocity 
        beta2 (float): additionally parameter for relaxtion rate of nematic order 
        chi0 (float): isotropic contractility 
        chi1 (float): anisotropic contractility
        Qsq (float): spontaneous nematic order parameter (if Qsq > 0, nematic order is spontaneous)
        """
        self.width = width
        self.height = height
        self.maxh = maxh
        
        # Physical parameters
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
        # Initialize simulation components
        self._create_mesh()
        self._setup_function_spaces()
        self._setup_initial_conditions()
        
    def _create_mesh(self):
        """Create the rectangular mesh with named boundaries."""
        shape = Rectangle(self.width, self.height).Face()
        shape.edges.Min(X).name = "right"
        shape.edges.Max(X).name = "left"
        shape.edges.Min(Y).name = "up"
        shape.edges.Max(Y).name = "down"
        self.mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=self.maxh)).Curve(3)
        
    def _setup_function_spaces(self):
        """Set up the function spaces for velocity, nematic order and density fields."""
        self.V = VectorH1(self.mesh, order=2, dirichlet="right|left|up|down")  # velocity field
        self.R = H1(self.mesh, order=2)  # density field

        self.Q = H1(self.mesh, order=2) #Qxx
        self.q = H1(self.mesh, order=2) #Qxy

        self.X = self.V * self.R * self.Q * self.q
        
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.nematic_xx, self.nematic_yx= self.gfu.components
        
    def _setup_initial_conditions(self):
        """Set up initial conditions for velocity and density fields."""
        # Default initial conditions - can be overridden
        self.density.Set(1)
        
        v = CoefficientFunction((0, 0))
        self.velocity.Set(v)

        self.nematic_xx.Set(0)
        self.nematic_yx.Set(0)

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test,Q_test, q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * InnerProduct(v_trial, v_test) * dx + 
                     self.eta_1 * InnerProduct(grad(v_trial), grad(v_test)) * dx +
                     self.eta_2 * div (v_trial) * div(v_test) * dx) 
                    #  + self.chi1 * (Q_trial * grad(v_test)[0,0] + q_trial *grad(v_test)[0,1])* dx   
                    #  + self.chi1 * (-Q_trial * grad(v_test)[1,1] + q_trial * grad(v_test)[1,0]) * dx )
        
        # Density evolution equation
        lhs_rho_time = (self.k * rho_trial * rho_test * dx + 
                       self.D * InnerProduct(grad(rho_trial), grad(rho_test)) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* InnerProduct(grad(Q_trial), grad(Q_test)) * dx + (- self.Qsq/ self.beta2 * Q_trial) * Q_test * dx  +
                  self.beta1/2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        
        q_time = ((self.kappa/self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx + (- self.Qsq / self.beta2 * q_trial) * q_test * dx  +
                  self.beta1/2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        
        return force_bal + lhs_rho_time + Q_time + q_time
    
    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test,Q_test, q_test) = functions 

        # advection term for density 
        advection_rho = (InnerProduct(grad(rho_trial), v_trial) * rho_test * dx + 
                          rho_trial * div(v_trial) * rho_test * dx)
        
        # nonlinear term in force balance equation 
        force_bal = (self.chi0 * rho_trial * div(v_test) / (rho_trial + 1) / 2 * dx
                     + self.chi1 * rho_trial * (Q_trial * grad(v_test)[0,0] + q_trial *grad(v_test)[0,1]) * dx  
                     + self.chi1 * rho_trial * (-Q_trial * grad(v_test)[1,1] + q_trial * grad(v_test)[1,0]) * dx) 

        # advection terms of nematic order 
        Q_adv = (InnerProduct(grad(Q_trial), v_trial) * Q_test * dx
                 + (grad(v_trial)[0,1] - grad(v_trial)[1, 0]) * q_trial * Q_test * dx) 
        
        q_adv = (InnerProduct(grad(q_trial), v_trial) * q_test * dx
                 - (grad(v_trial)[0,1] - grad(v_trial)[1, 0]) * Q_trial * q_test * dx)
        
        Q_hot = 1/self.beta2 * Q_test * Q_trial * Q_trial * Q_trial * dx 
        q_hot = 1/self.beta2 * q_test * q_trial * q_trial * q_trial * dx 
        
        return advection_rho + force_bal + q_adv + Q_hot + q_hot  + Q_adv
    
    def _setup_linear_form(self, functions): 
        _, (_,rho_test,_, _) = functions 
        return self.k * rho_test * dx 
    
    def _setup_inverse_form(self, functions, bilinear, tau): 
        (_, rho_trial, Q_trial, q_trial), (_,rho_test,Q_test, q_test) = functions 
        inverse_form = (rho_trial * rho_test * dx
                        + Q_trial * Q_test * dx
                        + q_trial * q_test * dx + tau * bilinear)
        return inverse_form

        
    def _setup_forms(self, tau):
        """Set up the bilinear and linear forms for the simulation."""
        functions = self.X.TnT()
        
        # set up bilinear forms 
        bilinear = self._setup_bilinear_form(functions)
        self.a = BilinearForm(self.X)
        self.a += bilinear 
        self.a.Assemble()
        
        # Nonlinear terms
        self.nonlinear = BilinearForm(self.X, nonassemble=True)
        self.nonlinear += self._setup_nonlinear_form(functions)
        
        # Linear terms, so that rho = 1 in steady state 
        self.f = LinearForm(self.X)
        self.f += self._setup_linear_form(functions)
        self.f.Assemble()

        # inverse form 
        mstar = BilinearForm(self.X)
        mstar += self._setup_inverse_form(functions, bilinear, tau)
        mstar.Assemble() 
        self.inv = mstar.mat.Inverse(freedofs= self.X.FreeDofs(),inverse="sparsecholesky")
        
    def set_initial_density(self, density_function):
        """
        Set custom initial density distribution.
        
        Parameters:
        density_function: NGSolve CoefficientFunction defining initial density
        """
        self.density.Set(density_function)
        
    def set_initial_velocity(self, velocity_function):
        
        """
        Set custom initial velocity field.
        
        Parameters:
        velocity_function: NGSolve CoefficientFunction defining initial velocity
        """
        self.velocity.Set(velocity_function)
        
    def simulate(self, tend=10, tau=0.01, save_interval=1):
        """
        Run the simulation.
        
        Parameters:
        tend (float): End time of simulation
        tau (float): Time step size
        save_interval (int): Interval for saving results (in steps)
        
        Returns:
        tuple: Lists of saved density and velocity fields
        """
        t = 0
        i = 0
        
        self._setup_forms(tau)
        
        # For storing results
        rhos = []
        velocities = []
        times = []
        
        # Create multidimensional GridFunction for animation
        self.gfut = GridFunction(self.gfu.space, multidim=0)
        
        with TaskManager():
            for i in tqdm(range(int(tend/tau))):
                self.nonlinear.Assemble()
                res = (self.a.mat * self.gfu.vec + 
                      self.nonlinear.mat * self.gfu.vec - 
                      self.f.vec)
                
                self.gfu.vec.data -= tau * self.inv * res

                if i % save_interval == 0:                    
                    self.gfut.AddMultiDimComponent(self.gfu.vec)                
                t += tau
        
        return times, rhos, velocities
    
    def visualize(self, animate=True):
        """
        Visualize the simulation results.
        
        Parameters:
        animate (bool): If True, show animation of results
        """
        if animate and hasattr(self, 'gfut'):
            Draw(self.gfut.components[0], self.mesh, 
                 interpolate_multidim=True, animate=True,
                 autoscale=True, vectors=True)
            Draw(self.gfut.components[1], self.mesh, 
                 interpolate_multidim=True, animate=True,
                 autoscale=True)
            Draw(self.gfut.components[2], self.mesh, 
                 interpolate_multidim=True, animate=True,
                 autoscale=True)
        else:
            Draw(self.velocity, self.mesh, "velocity", vectors=True)
            Draw(self.density, self.mesh, "density")
            Draw(self.nematic_xx, self.mesh, "nematic_xx")
            Draw(self.nematic_yx, self.mesh, "nematic_yx")
     
           