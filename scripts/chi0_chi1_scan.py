#!/usr/bin/env python3
"""
Grid scan of chi0 and chi1 for MyosinActinGel2D model.
Saves the entire time evolution every t=1 for T=30.
"""

import numpy as np
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import importlib
import utils
import myosin_actin_gel_fem

importlib.reload(utils)
importlib.reload(myosin_actin_gel_fem)

from myosin_actin_gel_fem import MyosinActinGel2D
from utils import tanh_source

overwrite = True # Set to False to skip existing files

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent.parent / "sim_data" / "acto_myo"
output_dir.mkdir(parents=True, exist_ok=True)

# Simulation parameters (from actin_myo_gel.ipynb)
T   = 100
tau = 0.05
save_interval = int(1.0 / tau)  # Save every t=1 (20 time steps)

# Grid of chi0 and chi1 values
n_chi = 50
chi0_values = np.exp(np.linspace(np.log(0.01), np.log(1), n_chi))  # Logarithmic spacing between 0.01 and 1
chi1_values = np.exp(np.linspace(np.log(0.01), np.log(1), n_chi))  # Logarithmic spacing between 0.01 and 1

# Fixed source fields (from actin_myo_gel.ipynb 2D example)
width = 0.05
center = 0.5
interfacial_width = 0.005
Lx = 1
Ly = 1
source = tanh_source(center=center, width=width, value=1, axis='x', interface_length=interfacial_width) * \
         tanh_source(center=center, width=Ly*0.9, value=1, axis='y', interface_length=0.1)

S = 0.8*source + 0.2
m0 = source
Qsq = 2*source - 1

# Other fixed parameters (from actin_myo_gel.ipynb 2D example)
params = {
    'width': 1, 'height': 1, 'maxh': 0.03,
    # mechanics
    'gamma': 1,
    'eta_1': 1,
    'eta_2': 0,
    # nematic
    'kappa': 1e-4,
    'beta1': 0.2,
    'beta2': 1,
    'Qsq': Qsq,
    # density / myosin
    'S': S,
    'm0': m0,
    'k0': 0.2,
    'k1': 0.8,
    'k_m': 0.5,
    'n_hill': 4,
    'm_ref': 0.12,
    'D_rho': 1e-4,
    'D_m': 1e-4,
    # base
    'k': 0.5,
    'D': 1e-3,
    'rho0': 1.0,
}

# Run grid scan
print(f"Running grid scan: {n_chi} x {n_chi} = {n_chi**2} simulations")
print(f"Final time: T={T}, save interval: {save_interval} time steps (dt=1.0)")
print(f"Output directory: {output_dir}\n")

count = 0
for i, chi0 in enumerate(chi0_values):
    for j, chi1 in enumerate(chi1_values):
        count += 1
        filename = output_dir / f"chi0_{chi0:.4f}_chi1_{chi1:.4f}.npz"

        if filename.exists() and not overwrite:
            print(f"[{count}/{n_chi**2}] Skipping chi0={chi0:.4f}, chi1={chi1:.4f} (file exists)")
            continue

        print(f"[{count}/{n_chi**2}] Running chi0={chi0:.4f}, chi1={chi1:.4f}...", end='', flush=True)

        try:
            sim = MyosinActinGel2D(
                chi0=chi0,
                chi1=chi1,
                **params
            )

            sim.simulate(tend=T, tau=tau, save_interval=save_interval)
            sim.export(filename=str(filename), n_samples=60)
            print(" ✓")
        except Exception as e:
            print(f" ✗ (Error: {e})")

print(f"\nGrid scan complete! Data saved to {output_dir}")
