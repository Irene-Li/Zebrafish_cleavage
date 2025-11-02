## System requirements
Python 3 and jupyter-notebook. 
The following packages are needed: pandas, numpy, matplotlib, scipy, skimage, seaborn, ngsolve.

## Installation
Clone the github repository after installing all the required packages. 
Currently the repository is not set up as a python module. All the source codes are in the /src folder. 

## Demo and instructions for use 
See the notebooks folder for codes used to produce simulations and graphs in this paper: 

> Xin Tong, Yuting I. Li, Joséphine Schelle, Edouard Hannezo, Carl-Philipp Heisenberg \
> Non-canonical cytokinesis driven by mechanical uncoupling via nematic flows and adhesion-based invagination \
> bioRxiv 2025.10.15.682552; doi: https://doi.org/10.1101/2025.10.15.682552


Here is a detailed list of notebook contents. 

**Active gel theory** 
- `active_gel_fem.ipynb` contains codes for running isotropic active gel model (see ST of the paper). 
- `double_actin_gel.ipynb` contains code for running two component nematic active gel model, as used in the paper to describe actomyosin dynamics in Wild-type. 
- `double_actin_gel_carhoA.ipynb` is for CaRhoA perturbation. 
- `double_actin_gel_param_scan.ipynb` is for plotting the phase diagram.
- `unsaturated_monomer_model.ipynb` contains an extended model showing that even if we are not in the limit where actin monomer is saturated, the conclusions of the paper do not change qualitatively. 

**Constriction dynamics** 
- `constriction1.ipynb` is for the constriction dynamics in phase 1.
- `constriction2.ipynb` is for the constriction dynamics in phase 2.

**Data processing and analysis** 
- `furrow_height_data.ipynb` is the prototype code for developping furrow detection from heights. However, the figures in the paper are produced by codes from Xin (code available upon request to Xin).
- `furrow_width.ipynb` is for measuring furrow width from experimental kymographs. 
  
