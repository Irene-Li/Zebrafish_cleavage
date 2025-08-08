import numpy as np 
from jax import numpy as jnp
import jax 
import diffrax 
import optax
from jax import jit, vmap

def F(theta): 
    return (1 + 3/2*jnp.cos(theta) - 1/2*(jnp.cos(theta))**3)**(1/3)

def Delta(theta, phi): 
    return jnp.clip(jnp.sin(theta) - F(theta)*jnp.cos(phi), 1e-5, None)

def D(theta, phi): 
    return F(theta)*jnp.sin(phi) 

def R_tilde(theta, phi): 
    return (D(theta, phi)**2 + Delta(theta, phi)**2)/(2*Delta(theta, phi))

def psi(theta, phi): 
    return jnp.arcsin(D(theta, phi)/R_tilde(theta, phi))

def zeta(theta, phi): 
    return 2*psi(theta, phi)*R_tilde(theta, phi)/F(theta)/np.pi 

def h(theta): 
    return jnp.sin(theta)/ F(theta)

def G(theta, phi, k): 
    term1 = (1+jnp.cos(theta))/F(theta)**2
    term2 = k*zeta(theta, phi) 
    return term1 + term2 

def total_energy(theta, phi, k):
    return G(theta, phi, k) - 2*phi/np.pi*k


dzeta_dtheta = jax.grad(zeta, argnums=0)
dzeta_dphi = jax.grad(zeta, argnums=1)
dG_dtheta = jax.grad(G, argnums=0)
ddG = jax.grad(dG_dtheta, argnums=0)
dG_dphi = jax.grad(G, argnums=1)
dh_dtheta = jax.grad(h)

def dtheta_dt(theta, phi, params):
    k, Ta, l, phi_dot = params
    a = Ta/2*(1+jnp.cos(theta) + l*zeta(theta, phi)/h(theta)**2)*dh_dtheta(theta)**2
    b = dG_dtheta(theta, phi, k) 
    c = (dG_dphi(theta, phi, k) - 2*k/np.pi)*phi_dot*(1 - jnp.tanh(100*(phi-np.pi/2*0.9)))/2

    a = jnp.where(jnp.abs(a) < 1e-10, 1e-10, a)

    discriminant = b**2 - 4*a*c
    root1 = (-jnp.sqrt(discriminant) - b)/(2*a)
    root2 = (jnp.sqrt(discriminant) - b)/(2*a)
    desired_sign = -b
    return jnp.where(root1 * desired_sign > 0, root1, root2)

# ======================================================================================================================
# phase 2 functions 
# ======================================================================================================================

def area_under_arc(hf_val, R_tilde_val, F_val):
    psi_val = jnp.arcsin(F_val/R_tilde_val)
    total_area = psi_val*R_tilde_val**2
    triangle_area = F_val*(R_tilde_val - hf_val)
    return (total_area - triangle_area)/F_val**2

def contact_area(theta0, hf): 
    A1 = area_under_arc(jnp.sin(theta0), R_tilde(theta0, np.pi/2), F(theta0))
    R_tilde2 = (F(theta0)**2 + hf**2)/(2*hf)
    A2 = area_under_arc(hf, R_tilde2, F(theta0))
    return A1 - A2

def arc_length2(theta0, hf): 
    R_tilde2 =  (F(theta0)**2 + hf**2)/(2*hf)
    psi_val = jnp.arcsin(F(theta0)/R_tilde2)
    return 2*psi_val*R_tilde2/F(theta0)/np.pi 

def total_energy2(theta0, hf, k, nu): 
    term1 = contact_area(theta0, hf)*nu/(2*np.pi)*2 # because there are two surfaces 
    term2 = k*arc_length2(theta0, hf)
    term3 = (1+jnp.cos(theta0))/F(theta0)**2
    source_term = - k 
    return term1 + term2 + term3 + source_term