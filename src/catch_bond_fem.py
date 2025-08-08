from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D,  NematicActiveGel1D, ActiveGelCircle

tanh = lambda arg: (exp(arg) - exp(-arg))/(exp(arg)+exp(-arg)) 

class CatchBond1D(NematicActiveGel1D):

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial), (v_test,rho_test,Q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * v_trial * v_test * dx + 
                    self.eta_1 * grad(v_trial) * grad(v_test) * dx +
                    self.eta_2 * grad(v_trial) * grad(v_test) * dx) 
        
        # Density evolution equation
        lhs_rho_time = (self.k * rho_trial * rho_test * dx + 
                       self.D * grad(rho_trial) * grad(rho_test) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* grad(Q_trial) * grad(Q_test) * dx 
                  + 1 / self.beta2 * Q_trial * Q_test * dx  
                  + self.beta1/2 * v_trial * grad(Q_test) * dx)
        
        return force_bal + lhs_rho_time + Q_time
    

    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial), (v_test,rho_test,Q_test) = functions 

        # advection term for density 
        advection_rho = (grad(rho_trial)*v_trial * rho_test * dx + 
                          rho_trial * grad(v_trial) * rho_test * dx)
        
        # nonlinear term in force balance equation 
        force_bal = (self.chi0 * rho_trial * grad(v_test) * 2 / (rho_trial + 1) * dx
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial * grad(v_test) * dx ) 

        # advection terms of nematic order 
        Q_adv = grad(Q_trial) * v_trial * Q_test * dx

        T = self.eta_1*(grad(v_trial))/2 + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial
        Q_cb = - 1/self.beta2 * tanh(T/0.1) * Q_test * self.Qsq(self.time) * dx 
                
        return advection_rho + force_bal + Q_adv + Q_cb
    


class CatchBond2D(NematicActiveGel2D):

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test,Q_test, q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * InnerProduct(v_trial, v_test) * dx + 
                    self.eta_1*InnerProduct(grad(v_trial), grad(v_test))* dx 
                    + self.eta_2*div(v_trial) * div(v_test) * dx)
        
        # Density evolution equation
        lhs_rho_time = (self.k * rho_trial * rho_test * dx + 
                       self.D * InnerProduct(grad(rho_trial), grad(rho_test)) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* InnerProduct(grad(Q_trial), grad(Q_test)) * dx +
                   (1/ self.beta2 * Q_trial) * Q_test * dx  +
                  self.beta1/2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        
        q_time = ((self.kappa/self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx + 
                  (1/ self.beta2 * q_trial) * q_test * dx  +
                  self.beta1/2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        
        return force_bal + lhs_rho_time + Q_time + q_time
    
    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test,Q_test, q_test) = functions 

        # advection term for density 
        advection_rho = (InnerProduct(grad(rho_trial), v_trial) * rho_test * dx + 
                          rho_trial * div(v_trial) * rho_test * dx)
        
        # nonlinear term in force balance equation 
        force_bal = (self.chi0 * rho_trial * div(v_test) * 2 / (rho_trial + 1) * dx
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (Q_trial * grad(v_test)[0,0] + q_trial *grad(v_test)[1,0]) * dx  
                     + self.chi1 * rho_trial * 2 / (rho_trial + 1) * (-Q_trial * grad(v_test)[1,1] + q_trial * grad(v_test)[0,1]) * dx) 

        # advection terms of nematic order 
        Q_adv = (InnerProduct(grad(Q_trial), v_trial) * Q_test * dx
                 + (grad(v_trial)[1, 0] - grad(v_trial)[0, 1]) * q_trial * Q_test * dx) 
        
        q_adv = (InnerProduct(grad(q_trial), v_trial) * q_test * dx
                 - (grad(v_trial)[1,0] - grad(v_trial)[0, 1]) * Q_trial * q_test * dx)
        
        Txx = self.eta_1*(grad(v_trial)[0, 0] - grad(v_trial)[1, 1])/2 + self.chi1 * rho_trial * 2 / (rho_trial + 1) * Q_trial 
        Txy = self.eta_1*(grad(v_trial)[0, 1] + grad(v_trial)[1, 0])/2 + self.chi1 * rho_trial * 2 / (rho_trial + 1) * q_trial
        
        Q_hot = - 1/self.beta2 * Q_test * tanh(Txx/0.1) * self.Qsq(self.time) * dx  
        q_hot = - 1/self.beta2 * q_test * tanh(Txy/0.1) * self.Qsq(self.time) * dx 
        
        return advection_rho + force_bal + Q_hot + q_hot + Q_adv + q_adv