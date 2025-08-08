from ngsolve import *
from ngsolve.meshes import Make1DMesh
from netgen.occ import *
import numpy as np
from ngsolve.webgui import Draw
from tqdm import tqdm
import matplotlib.pyplot as plt
from active_gel_fem import NematicActiveGel2D,  NematicActiveGel1D

class NonconsPhaseSeparation2D(NematicActiveGel2D):

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test,Q_test, q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * InnerProduct(v_trial, v_test) * dx + 
                    self.eta_1*InnerProduct(grad(v_trial), grad(v_test))* dx 
                    + self.eta_2*div(v_trial) * div(v_test) * dx)
        
        # Density evolution equation
        lhs_rho_time = (self.k * 0.5* self.rho0* rho_trial * rho_test * dx + 
                       self.D * InnerProduct(grad(rho_trial), grad(rho_test)) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* InnerProduct(grad(Q_trial), grad(Q_test)) * dx +
                   (- self.Qsq / self.beta2 * Q_trial) * Q_test * dx  +
                  self.beta1/2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        
        q_time = ((self.kappa/self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx + 
                  (- self.Qsq/ self.beta2 * q_trial) * q_test * dx  +
                  self.beta1/2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        
        return force_bal + lhs_rho_time + Q_time + q_time
    
    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test, Q_test, q_test) = functions 

        # advection term for density 
        advection_rho = (InnerProduct(grad(rho_trial), v_trial) * rho_test * dx + 
                          rho_trial * div(v_trial) * rho_test * dx)
        
        # nonlinear term in density evolution equation
        rho_hot = self.k * rho_test * (rho_trial*rho_trial*rho_trial - (self.rho0 + 0.5) *rho_trial) * dx 
        
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
        
        return advection_rho + force_bal + Q_hot + q_hot + Q_adv + q_adv + rho_hot 
    
    def _setup_linear_form(self, functions): 
        _, (_,rho_test, _, _) = functions 
        l1 = 0 * rho_test * dx 
        return l1 
    
class ConsPhaseSeparation2D(NematicActiveGel2D):

    def _setup_bilinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test,Q_test, q_test) = functions 

        # Force balance equation
        force_bal = (self.gamma * InnerProduct(v_trial, v_test) * dx + 
                    self.eta_1*InnerProduct(grad(v_trial), grad(v_test))* dx 
                    + self.eta_2*div(v_trial) * div(v_test) * dx)
        
        # Density evolution equation
        aux_trial = rho_trial.Operator('hesse')
        aux_test = rho_test.Operator('hesse')
        lhs_rho_time = (self.k * rho_trial * rho_test * dx + 
                       self.D * ( 0.5 * InnerProduct(grad(rho_trial), grad(rho_test)) 
                                 + self.kappa * (aux_trial[0, 0] + aux_trial[1, 1]) * (aux_test[0, 0] + aux_test[1, 1])) * dx)
        
        #nematic order evolution equation                              
        Q_time = ((self.kappa/self.beta2)* InnerProduct(grad(Q_trial), grad(Q_test)) * dx +
                   (- self.Qsq / self.beta2 * Q_trial) * Q_test * dx  +
                  self.beta1/2 * (v_trial[0] * grad(Q_test)[0] - v_trial[1] * grad(Q_test)[1]) * dx)
        
        q_time = ((self.kappa/self.beta2) * InnerProduct(grad(q_trial), grad(q_test)) * dx + 
                  (- self.Qsq/ self.beta2 * q_trial) * q_test * dx  +
                  self.beta1/2 * (v_trial[1] * grad(q_test)[0] + v_trial[0] * grad(q_test)[1]) * dx)
        
        return force_bal + lhs_rho_time + Q_time + q_time
    
    def _setup_nonlinear_form(self, functions): 
        (v_trial, rho_trial, Q_trial, q_trial), (v_test,rho_test, Q_test, q_test) = functions 

        # advection term for density 
        advection_rho = (InnerProduct(grad(rho_trial), v_trial) * rho_test * dx + 
                          rho_trial * div(v_trial) * rho_test * dx)
        
        # nonlinear term in density evolution equation
        rho_hot = self.D * InnerProduct(grad(rho_test), (grad(rho_trial*rho_trial*rho_trial)-1.5*grad(rho_trial*rho_trial))) * dx 
        
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
        
        return advection_rho + force_bal + Q_hot + q_hot + Q_adv + q_adv + rho_hot 
    
    def _setup_linear_form(self, functions): 
        _, (_,rho_test, _, _) = functions 
        l1 = self.k * self.rho0 * rho_test * dx 
        return l1 