# Physics informed learning of phase field equations
This repository contains codes for physics informed learning of phase field equations. One dimensional examples of allen cahn and cahn hilliard equations are presented here. Unlike the traditional approach of PINN's where the parameters of the equation are learned from gradient descent, here the parameters of the equation are approximated using least squares fit at every training epoch. For the purpose of comparison the codes where the equation parameters are learned by traditional gradient descent are also presented.


## Getting started
The first step is to install anaconda in your system
and create the conda enviroment necessary to run the above codes.

```
conda create env -f sciml_gpu.yaml
cond activate sciml_gpu
```
codes where least squares fit is used to learn equation parameters
```
python PINN_AC_1D_LSfit.py
python PINN_CH_1D_LSfit.py
```
codes where gradient descent is used to learn equation parameters

```
python PINN_AC_1D.py
python PINN_CH_1D.py
```

All the plots and data files will be automatically stored in the corresponsing results folder.


