import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


class ActiveGel():
    
    def __init__(self, b1, b2, chi, kappa, etas, xi, zeta, k, gamma, source):
        # Initialises the class with the model parameters 
        self.b1 = b1 
        self.b2 = b2
        self.chi = chi 
        self.kappa = kappa 
        self.eta0 = etas[0]
        self.eta1 = etas[1]
        self.xi = xi 
        self.zeta = zeta 
        self.k = k
        self.gamma = gamma 
        self.source = source 

         
    def initialise(self, L, T, n_frames, init=None, seed=None): 
        # Set up the simulation parameters 
        self.L = int(L) 
        self.size = int(L*2)
        self.T = T 
        self.n_frames = int(n_frames)
        if init is not None: 
            self.initial_state = init 
        else: 
            self.initial_state = np.zeros((self.size))
        self._set_M()

    def evolve(self):
        # The core function that integrates the ODEs forward. 
        t_span = np.linspace(0, self.T, self.n_frames)
        self.res = solve_ivp(self._rhs, (0, self.T), self.initial_state, 
                             t_eval=t_span, method='LSODA', rtol=1e-5, min_step=1e-8)
        evo = self.res.y.reshape((2, self.L, self.n_frames))
        self.time = t_span 
        self.Q, self.rho = np.rollaxis(evo, -1, 1)
        self.v = np.array(list(map(self._solve_for_v, t_span, self.rho, self.Q)))
        
    def _set_M(self):
        kx = np.fft.fftfreq(self.L)*2*np.pi
        Mxx = - (self.eta0+self.eta1)*kx*kx - self.gamma # compression, screened by friction  
        self.M = 1/Mxx
        self.ik = 1j*kx

    def _solve_for_v(self, t, rho, Q): 
        P = self.xi*self.source(t)*rho*4/(1+rho)
        Pk = np.fft.fft(P)
        
        Qk = np.fft.fft(Q*rho*2/(1+rho))
        # coupling, ik P + ik_j Q_{ij} 
        
        t1 = self.ik*Pk
        t2 = self.zeta*self.ik*Qk 
        vk = self.M*(t1+t2)
        v = np.fft.ifft(vk).real

        return v
                
    def _rhs(self, t, y):
        Q, rho = y.reshape((2, self.L))
                
        # solve for v
        vx = self._solve_for_v(t, rho, Q)
        vxx = self._dx(vx)
        lapQ = self._laplacian(Q)
        
        # rho equation 
        drho = self._mass_flow(vx, rho) - self.k*rho*(rho-1) + self._laplacian(rho)
        
        # Q equations 
        dQ = self.b1/2*(vxx) -self.chi/self.b2*Q + self.kappa/self.b2*lapQ # force 
        dQ -= self._advection(vx, Q)  
        return np.stack([dQ, drho]).flatten()
        
    def _dx(self, f):
        return (np.roll(f, -1) - np.roll(f, 1))/2 
        
  
    def _laplacian(self, Q): 
        lapQ = np.roll(Q, 1) + np.roll(Q, -1) - 2*Q 
        return lapQ 
    
    def _mass_flow(self, v, rho): 
        diff = - np.abs(v)*rho # flow of mass out of the lattice 
        for i in [-1, 1]: 
            v_neighbour = np.roll(v, i) 
            diff[i*v_neighbour > 0] += (np.abs(v_neighbour)*np.roll(rho, i))[i*v_neighbour > 0 ] 
        return diff 
    
    def _advection(self, v, f): 
        a1 = np.min(v, 0) 
        a2 = np.max(v, 0)  
        return a2*(f - np.roll(f, 1)) + a1*(np.roll(f, -1) - f) 
        
class ActiveGel2D(ActiveGel):

         
    def initialise(self, L, T, n_frames, init=None, seed=None): 
        # Set up the simulation parameters 
        super().initialise(L, T, n_frames, init, seed)
        self.size = int(L*L*3)
        if init is None: 
            self.initial_state = np.zeros((self.size))

    def evolve(self):
        # The core function that integrates the ODEs forward. 
        t_span = np.linspace(0, self.T, self.n_frames)
        self.res = solve_ivp(self._rhs, (0, self.T), self.initial_state, t_eval=t_span, method='RK45', rtol=1e-5)
        evo = self.res.y.reshape((3, self.L, self.L, self.n_frames))
        self.Q, self.q, self.rho = np.rollaxis(evo, -1, 1)
        self.v = np.array(list(map(self._solve_for_v, t_span, self.rho, self.Q, self.q)))
        
    def _set_M(self):
        k = np.fft.fftfreq(self.L)*2*np.pi
        kx, ky = np.meshgrid(k, k, indexing='ij')
        Mxx = (self.eta0+self.eta1)*kx*kx + self.eta0*ky*ky + self.gamma # compression, screened by friction  
        Mxy = self.eta1*kx*ky  
        Myy = (self.eta0+self.eta1)*ky*ky + self.eta0*kx*kx + self.gamma # compression, screened by friction 
        A = np.stack([Mxx, Mxy, Mxy, Myy], axis=-1).reshape((self.L, self.L, 2, 2)).astype('complex')
        self.M = np.linalg.inv(A)
        self.ik = 1j*np.stack([kx, ky], axis=-1)

    def _solve_for_v(self, t, rho, Q, q): 
        P = self.xi*self.source(t)*rho*4/(1+rho)
        Pk = np.fft.fft2(P)
        
        Qk = np.fft.fft2(Q*rho*2/(1+rho))
        qk = np.fft.fft2(q*rho*2/(1+rho)) 
        
        # coupling, ik P + ik_j Q_{ij} 
        
        t1 = self.ik*Pk[:, :, np.newaxis]
        t2 = self.ik[:, :, 0]*Qk + self.ik[:, :, 1]*qk 
        t3 = self.ik[:, :, 0]*qk - self.ik[:, :, 1]*Qk
        t4 = self.zeta*np.stack([t2, t3], axis=(-1))

        
        vk = np.einsum('ijkl,ijl->ijk', self.M, (t1+t4))
        vx = np.fft.ifft2(vk[:, :, 0]).real
        vy = np.fft.ifft2(vk[:, :, 1]).real
        return vx, vy
                
    def _rhs(self, t, y):
        Q, q, rho = y.reshape((3, self.L, self.L))
        
        Qx = self._dx(Q)
        Qy = self._dy(Q)
        qx = self._dx(q)
        qy = self._dy(q)
        
        # solve for v
        vx, vy = self._solve_for_v(t, rho, Q, q)
        
        vxx = self._dx(vx)
        vyy = self._dy(vy)
        vxy = self._dx(vy)
        vyx = self._dy(vx)
        exy = (vxy + vyx)/2
        lapQ = self._laplacian(Q)
        lapq = self._laplacian(q)

        # rho equation 
        drho = self._mass_flow(vx, vy, rho) - self.k*rho*(rho-1) + self._laplacian(rho)
        
        # Q equations 
        dQ = self.b1/2*(vxx - vyy) -self.chi/self.b2*Q + self.kappa/self.b2*lapQ # force 
        dq = self.b1*exy - self.chi/self.b2*q + self.kappa/self.b2*lapq 
        dQ -= self._advection(vx, Q, 0) + self._advection(vy, Q, 1) + (vxy-vyx)/2*q # co derivatives 
        dq -= self._advection(vx, q, 0) + self._advection(vy, q, 1) - (vxy-vyx)*Q
        return np.stack([dQ, dq, drho]).flatten()
        
    def _dx(self, f): # midpoint discretisation 
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/2 
    
    def _dy(self, f): # midpoint discretisation 
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/2
        
  
    def _laplacian(self, Q): 
        lapQ = np.zeros_like(Q)     
        for axis in [0, 1]:
            lapQ += np.roll(Q, 1, axis=axis) + np.roll(Q, -1, axis=axis) - 2*Q 
        return lapQ 
    
    
    def _mass_flow(self, vx, vy, rho): 
        diff = - (np.abs(vx)+np.abs(vy))*rho # flow of mass out of the lattice 
        for (a, v) in enumerate([vx, vy]): 
            for i in [-1, 1]: 
                v_neighbour = np.roll(v, i, axis=a) 
                diff[i*v_neighbour > 0] += (np.abs(v_neighbour)*np.roll(rho, i, axis=a))[i*v_neighbour > 0 ] 
        return diff 
    
    def _advection(self, v, f, a): 
        a1 = np.min(v, 0) 
        a2 = np.max(v, 0)  
        return a2*(f - np.roll(f, 1, axis=a)) + a1*(np.roll(f, -1, axis=a) - f) 
