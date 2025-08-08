from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D,  NematicActiveGel1D, ActiveGelCircle


class DoubleActinGel2D(NematicActiveGel2D):
    def __init__(self, k_actin=1, rho0_actin=1, **kwargs):
        super().__init__(**kwargs)
        self.k_actin = k_actin
        self.rho0_actin = rho0_actin
        
    def _setup_function_spaces(self):
        """Set up the function spaces for velocity, nematic order and density fields."""
        self.V = VectorH1(self.mesh, order=2, dirichlet="right|left|up|down")  # velocity field
        self.R = H1(self.mesh, order=2)  # density field
        self.R2 = H1(self.mesh, order=2)  # density field for actin

        self.Q = H1(self.mesh, order=2) #Qxx
        self.q = H1(self.mesh, order=2) #Qxy

        self.X = self.V * self.R * self.R2 * self.Q * self.q
        
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.density2, self.nematic_xx, self.nematic_yx= self.gfu.components
        self.time = Parameter(0)
        
    def _setup_initial_conditions(self):
        """Set up initial conditions for velocity and density fields."""
        super()._setup_initial_conditions()
        self.density2.Set(1)

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial), (v_test,rho_test,rho2_test, Q_test, q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * InnerProduct(v_trial, v_test) * dx + 
                    self.eta_1*InnerProduct(grad(v_trial), grad(v_test))* dx 
                    + self.eta_2*div(v_trial) * div(v_test) * dx)
        
        # Density evolution equation
        lhs_rho_time = (self.k * rho_trial * rho_test * dx + 
                       self.D * InnerProduct(grad(rho_trial), grad(rho_test)) * dx)
        
        lhs_rho2_time = (self.k_actin * rho2_trial * rho2_test * dx +
                        self.D * InnerProduct(grad(rho2_trial), grad(rho2_test)) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* InnerProduct(grad(Q_trial), grad(Q_test)) * dx +
                   (-self.Qsq/self.beta2 * Q_trial) * Q_test * dx  +
                  self.beta1/2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        
        q_time = ((self.kappa/self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx + 
                  (-self.Qsq/self.beta2 * q_trial) * q_test * dx  +
                  self.beta1/2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        
        return force_bal + lhs_rho_time + Q_time + q_time + lhs_rho2_time
    
    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial), (v_test,rho_test, rho2_test, Q_test, q_test) = functions 

        # advection term for density 
        advection_rho = (InnerProduct(grad(rho_trial), v_trial) * rho_test * dx + 
                          rho_trial * div(v_trial) * rho_test * dx)
        
        advection_rho2 = (InnerProduct(grad(rho2_trial), v_trial) * rho2_test * dx +
                          rho2_trial * div(v_trial) * rho2_test * dx)
        
        # nonlinear term in force balance equation 
        force_bal = (self.chi0 * rho_trial * div(v_test) * 2 / (rho_trial + 1) * dx
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (Q_trial * grad(v_test)[0,0] + q_trial *grad(v_test)[1,0]) * dx  
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (-Q_trial * grad(v_test)[1,1] + q_trial * grad(v_test)[0,1]) * dx) 

        # advection terms of nematic order 
        Q_adv = (InnerProduct(grad(Q_trial), v_trial) * Q_test * dx
                 + (grad(v_trial)[1, 0] - grad(v_trial)[0, 1]) * q_trial * Q_test * dx) 
        
        q_adv = (InnerProduct(grad(q_trial), v_trial) * q_test * dx
                 - (grad(v_trial)[1,0] - grad(v_trial)[0, 1]) * Q_trial * q_test * dx)
        
        Q_hot = 1/self.beta2 * Q_test * (Q_trial * Q_trial + q_trial * q_trial) * Q_trial * dx  
        q_hot = 1/self.beta2 * q_test * (q_trial * q_trial + Q_trial * Q_trial) * q_trial * dx 
        
        return advection_rho + advection_rho2 + force_bal + Q_hot + q_hot + Q_adv + q_adv
    
    def _setup_linear_form(self, functions): 
        _, (_,rho_test, rho2_test, _, _) = functions 
        l1 = self.k * self.rho0*rho_test * dx 
        l2 = self.k_actin * self.rho0_actin * rho2_test * dx
        return l1 + l2
    
    def _setup_inverse_form(self, functions, bilinear, tau): 
        (_, rho_trial, rho2_trial, Q_trial, q_trial), (_,rho_test, rho2_test, Q_test, q_test) = functions 
        inverse_form = (rho_trial * rho_test * dx
                        + rho2_trial * rho2_test * dx
                        + Q_trial * Q_test * dx
                        + q_trial * q_test * dx + tau * bilinear)
        return inverse_form
    
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
            for i in range(1, 4):
                Draw(self.gfut.components[i], self.mesh, 
                     interpolate_multidim=True, animate=True,
                     autoscale=True)
        else:
            Draw(self.velocity, self.mesh, "velocity", vectors=True)
            Draw(self.density, self.mesh, "density")
            Draw(self.density2, self.mesh, "density")
            Draw(self.nematic_xx, self.mesh, "nematic_xx")
            Draw(self.nematic_yx, self.mesh, "nematic_yx")

    def export_to_npy(self, filename, n_samples=50):
        """
        Export simulation results to a .npy file.
        
        Parameters:
        filename (str): Name of the output file
        """
        if hasattr(self, 'gfut'):
            N = len(self.gfut.vecs)
            data = np.zeros((N, n_samples, n_samples, 6)) # 5 components: rho, v, Q, q
            rho_gf = GridFunction(self.R)
            rho2_gf = GridFunction(self.R2)
            v_gf = GridFunction(self.V)
            Q_gf = GridFunction(self.Q)
            q_gf = GridFunction(self.q)
            X = np.linspace(0, 1, n_samples)
            Y = np.linspace(0, 1, n_samples)
            for n in range(N): 
                rhos = self.gfut.components[1]
                rhos2 = self.gfut.components[2]
                v = self.gfut.components[0]
                Q = self.gfut.components[3]
                q = self.gfut.components[4]

                rho_gf.vec.data = rhos.vecs[n]
                v_gf.vec.data = v.vecs[n]
                Q_gf.vec.data = Q.vecs[n]
                q_gf.vec.data = q.vecs[n]
                rho2_gf.vec.data = rhos2.vecs[n]


                data[n, :, :, 0] = np.array([rho_gf(y, x) for x in X for y in Y]).reshape(n_samples, n_samples) # outer loop: x in X, inner loop: y in Y  
                data[n, :, :, 1] = np.array([rho2_gf(y, x) for x in X for y in Y]).reshape(n_samples, n_samples)
                data[n, :, :, 2:4] = np.array([v_gf(y, x) for x in X for y in Y]).reshape(n_samples, n_samples, 2)
                data[n, :, :, 4] = np.array([Q_gf(y, x) for x in X for y in Y]).reshape(n_samples, n_samples)
                data[n, :, :, -1] = np.array([q_gf(y, x) for x in X for y in Y]).reshape(n_samples, n_samples)
            np.save(filename, data)
            return data 
        else:
            raise ValueError("No simulation data to export. Run simulate() first.")
        
class UnsaturedMonomerModel(DoubleActinGel2D):
    def __init__(self, k2=0.1, k4=0.1, k5=0.1, **kwargs):
        super().__init__(**kwargs)
        self.k2 = k2 
        self.k4 = k4
        self.k5 = k5 

    def _setup_function_spaces(self):
        """Set up the function spaces for velocity, nematic order and density fields."""
        self.V = VectorH1(self.mesh, order=2, dirichlet="right|left|up|down")  # velocity field
        self.R = H1(self.mesh, order=2)  # density field
        self.R2 = H1(self.mesh, order=2)  # density field for actin
        self.R3 = H1(self.mesh, order=2)  # density field for monomers

        self.Q = H1(self.mesh, order=2) #Qxx
        self.q = H1(self.mesh, order=2) #Qxy

        self.X = self.V * self.R * self.R2 * self.Q * self.q * self.R3
        
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.density2, self.nematic_xx, self.nematic_yx, self.m = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        """Set up initial conditions for velocity and density fields."""
        super()._setup_initial_conditions()
        self.m.Set(1) 
        
    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial, m_trial), (v_test,rho_test,rho2_test, Q_test, q_test, m_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * InnerProduct(v_trial, v_test) * dx + 
                    self.eta_1*InnerProduct(grad(v_trial), grad(v_test))* dx 
                    + self.eta_2*div(v_trial) * div(v_test) * dx)
        
        # Density evolution equation
        lhs_rho_time = (self.k * rho_trial * rho_test * dx - self.k2 * m_trial * rho_test * dx + 
                       self.D * InnerProduct(grad(rho_trial), grad(rho_test)) * dx)
        
        lhs_rho2_time = (self.k_actin * rho2_trial * rho2_test * dx - self.k4 * m_trial * rho2_test * dx + 
                        self.D * InnerProduct(grad(rho2_trial), grad(rho2_test)) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* InnerProduct(grad(Q_trial), grad(Q_test)) * dx +
                   (-self.Qsq/self.beta2 * Q_trial) * Q_test * dx  +
                  self.beta1/2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        
        q_time = ((self.kappa/self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx + 
                  (-self.Qsq/self.beta2 * q_trial) * q_test * dx  +
                  self.beta1/2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        
        # Monomer evolution equation
        lhs_m_time = (- self.k * rho_trial - self.k_actin * rho2_trial 
                      + self.k2 * m_trial  + self.k4 * m_trial + self.k5 * m_trial) * m_test * dx 
        
        return force_bal + lhs_rho_time + Q_time + q_time + lhs_rho2_time + lhs_m_time 
    
    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, rho2_trial, Q_trial, q_trial, m_trial), (v_test,rho_test, rho2_test, Q_test, q_test, m_test) = functions 

        # advection term for density 
        advection_rho = (InnerProduct(grad(rho_trial), v_trial) * rho_test * dx + 
                          rho_trial * div(v_trial) * rho_test * dx)
        
        advection_rho2 = (InnerProduct(grad(rho2_trial), v_trial) * rho2_test * dx +
                          rho2_trial * div(v_trial) * rho2_test * dx)
        
        # nonlinear term in force balance equation 
        force_bal = (self.chi0 * rho_trial * div(v_test) * 2 / (rho_trial + 1) * dx
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (Q_trial * grad(v_test)[0,0] + q_trial *grad(v_test)[1,0]) * dx  
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (-Q_trial * grad(v_test)[1,1] + q_trial * grad(v_test)[0,1]) * dx) 

        # advection terms of nematic order 
        Q_adv = (InnerProduct(grad(Q_trial), v_trial) * Q_test * dx
                 + (grad(v_trial)[1, 0] - grad(v_trial)[0, 1]) * q_trial * Q_test * dx) 
        
        q_adv = (InnerProduct(grad(q_trial), v_trial) * q_test * dx
                 - (grad(v_trial)[1,0] - grad(v_trial)[0, 1]) * Q_trial * q_test * dx)
        
        Q_hot = 1/self.beta2 * Q_test * (Q_trial * Q_trial + q_trial * q_trial) * Q_trial * dx  
        q_hot = 1/self.beta2 * q_test * (q_trial * q_trial + Q_trial * Q_trial) * q_trial * dx 
        
        return advection_rho + advection_rho2 + force_bal + Q_hot + q_hot + Q_adv + q_adv

    def _setup_linear_form(self, functions): 
        _, (_,_,_, _, _, m_test) = functions 
        l3 = self.k5 * m_test * dx 
        return l3 
    
    def _setup_inverse_form(self, functions, bilinear, tau): 
        (_, rho_trial, rho2_trial, Q_trial, q_trial, m_trial), (_,rho_test, rho2_test, Q_test, q_test, m_test) = functions 
        inverse_form = (rho_trial * rho_test * dx + rho2_trial * rho2_test * dx + m_trial * m_test * dx 
                        + Q_trial * Q_test * dx
                        + q_trial * q_test * dx + tau * bilinear)
        return inverse_form
    
class DoubleFluidGel1D(DoubleActinGel2D): 

    def _create_mesh(self):
        self.mesh = Make1DMesh(int(1/self.maxh))

    def _setup_function_spaces(self):

        self.V = H1(self.mesh, order=2, dirichlet="right|left")  # velocity field
        self.R = H1(self.mesh, order=2)  # density field
        self.R2 = H1(self.mesh, order=2) # actin density
        self.Q = H1(self.mesh, order=2)  # Qxx

        self.X = self.V * self.R * self.R2 * self.Q
        
        self.gfu = GridFunction(self.X)
        self.velocity, self.density, self.density2, self.nematic_xx = self.gfu.components
        self.time = Parameter(0)

    def _setup_initial_conditions(self):
        """Set up initial conditions for velocity and density fields."""
        # Default initial conditions - can be overridden
        self.density.Set(0)
        self.density2.Set(1)
        self.velocity.Set(0)
        self.nematic_xx.Set(0)
        self.time.Set(0)

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, rho2_trial, Q_trial), (v_test,rho_test, rho2_test, Q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * v_trial * v_test * dx + 
                    self.eta_1 * grad(v_trial) * grad(v_test) * dx +
                    self.eta_2 * grad(v_trial) * grad(v_test) * dx) 
        
        # Density evolution equation
        lhs_rho_time = (self.k * rho_trial * rho_test * dx + 
                       self.D * grad(rho_trial) * grad(rho_test) * dx)
        
        lhs_rho2_time = (self.k_actin * rho2_trial * rho2_test * dx + 
                       self.D * grad(rho2_trial) * grad(rho2_test) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* grad(Q_trial) * grad(Q_test) * dx 
                  + (- self.Qsq / self.beta2 * Q_trial) * Q_test * dx  
                  + self.beta1/2 * v_trial * grad(Q_test) * dx)
        
        return force_bal + lhs_rho_time + Q_time + lhs_rho2_time
    

    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, rho2_trial, Q_trial), (v_test,rho_test, rho2_test, Q_test) = functions 

        # advection term for density 
        advection_rho = (grad(rho_trial) * v_trial * rho_test * dx + 
                          rho_trial * grad(v_trial) * rho_test * dx)
        
        advection_rho2 = (grad(rho2_trial) * v_trial * rho2_test * dx + 
                          rho2_trial * grad(v_trial) * rho2_test * dx)
        
        # nonlinear term in force balance equation 
        force_bal = (self.chi0 * rho_trial * grad(v_test) * 2 / (rho_trial + 1) * dx
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial * grad(v_test) * dx ) 

        # advection terms of nematic order 
        Q_adv = grad(Q_trial) * v_trial * Q_test * dx

        # higher order terms in the nematic equation 
        Q_hot = 1/self.beta2 * Q_test * Q_trial * Q_trial * Q_trial * dx  
                
        return advection_rho + force_bal+ Q_hot + Q_adv + advection_rho2
    
    def _setup_linear_form(self, functions): 
        _, (_,rho_test,rho2_test,_) = functions 
        l1 = self.k * self.rho0 * rho_test * dx 
        l2 = self.k_actin * self.rho0_actin * rho2_test * dx 
        return l1 + l2
    
    def _setup_inverse_form(self, functions, bilinear, tau): 
        (_, rho_trial, rho2_trial, Q_trial), (_,rho_test, rho2_test, Q_test) = functions 
        inverse_form = (rho_trial * rho_test * dx
                        + rho2_trial * rho2_test * dx
                        + Q_trial * Q_test * dx
                        + tau * bilinear)
        return inverse_form
    
    def export_to_npy(self, n_samples):
        """
        Visualize the simulation results.
        """
        if hasattr(self, 'gfut'):
            N = len(self.gfut.vecs)
            data = np.zeros((N, n_samples, 4)) # 4 components

            rho_gf = GridFunction(self.R) 
            rho2_gf = GridFunction(self.R2)
            v_gf = GridFunction(self.V) 
            Q_gf = GridFunction(self.Q) 
            X = np.linspace(0, 1, n_samples)

            for n in range(N): 
                rhos = self.gfut.components[1]
                rho2s = self.gfut.components[2]
                v = self.gfut.components[0]
                Q = self.gfut.components[-1]

                rho_gf.vec.data = rhos.vecs[n]
                rho2_gf.vec.data = rho2s.vecs[n]
                v_gf.vec.data = v.vecs[n]
                Q_gf.vec.data = Q.vecs[n]

                data[n,  :, 0] = np.array([rho_gf(x) for x in X])
                data[n, :, 1] = np.array([rho2_gf(x) for x in X])
                data[n,  :, 2] = np.array([v_gf(x) for x in X])
                data[n,  :, 3] = np.array([Q_gf(x) for x in X])         
            return data 
        else:
            raise ValueError("No simulation data to export. Run simulate() first.")
