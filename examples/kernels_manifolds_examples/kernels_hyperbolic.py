import numpy as np
import random
import torch
import gpytorch
import pymanopt.manifolds as pyman_man

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.pymanopt_addons.manifolds.hyperbolic import HyperbolicLorentz
from BoManifolds.Riemannian_utils.hyperbolic_utils_torch import lorentz_distance_torch
from BoManifolds.kernel_utils.kernels_hyperbolic import HyperbolicRiemannianMillsonGaussianKernel, \
    HyperbolicRiemannianMillsonIntegratedMaternKernel

# Use double precision
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

'''
This example shows the computation of various kernels on the hyperbolic Lorentz manifold. 
The kernels are computed between the starting point of a geodesic and points equally spaced along this geodesic.

The following kernels are considered:
 Geometry-aware kernels: 
  - HyperbolicRiemannianMillsonGaussianKernel [2]
  - HyperbolicRiemannianMillsonIntegratedMaternKernel [1] (allows the optimization of the smoothness parameter)
 Euclidean kernels:
  - RBFKernel
  - MaternKernel

The example displays the kernel value in function of the distance on the manifold for each of the considered kernels.

Note: the kernel parameters (lengthscale and smoothness) may not have the same impact depending on the kernels, i.e., 
Riemannian and Euclidean kernels are not expected to behave exactly similarly for the same parameters values.

References:
[1] N. Jaquier, V. Borovitskiy, A. Smolensky, A. Terenin, T. Asfour, and L. Rozo. 
Bayesian Optimization meets Riemannian Manifolds in Robot Learning. 
In Conference on Robot Learning, 2021. 

[2] A. Grigoryan and M. Noguchi. 
The heat kernel on hyperbolic space. 
Bulletin of the London Mathematical Society, 30(6):643–650, 1998.

This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
'''


if __name__ == "__main__":
    seed = 1234
    # Set numpy and pytorch seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the dimension
    dim_data = 4
    dim_hyperbolic = dim_data - 1  # We are on H^d embedded in R^d+1

    # Define the manifold
    manifold = HyperbolicLorentz(dim_hyperbolic)

    # Generate data
    nb_data = 100
    base = -5 * np.ones((dim_data, 1))
    base[0] = np.sqrt(1. + np.sum(base[1:, :] ** 2, 0))
    point = 10*np.ones((dim_data, 1))
    point[0] = np.sqrt(1. + np.sum(point[1:, :]**2, 0))
    u = manifold.log(base, point)
    data = np.array([manifold.exp(base, u, t/(nb_data-1)) for t in range(nb_data)])[:, :, 0]
    x1 = torch.from_numpy(data).to(device)
    x2 = torch.from_numpy(point.T).to(device)
    x2 = x1[-1][None]

    # Kernels parameters
    lengthscale_hyperbolic = 1.
    lengthscale_euclidean = 5.
    smoothness = 2.5

    # RBF kernel
    rbf_kernel = HyperbolicRiemannianMillsonGaussianKernel(dim_hyperbolic)
    rbf_kernel.lengthscale = lengthscale_hyperbolic
    rbf_kernel.to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Integrated Matérn kernel
    integrated_matern_kernel = HyperbolicRiemannianMillsonIntegratedMaternKernel(dim=dim_hyperbolic, nu=smoothness)
    integrated_matern_kernel.lengthscale = lengthscale_hyperbolic
    integrated_matern_kernel.to(device)
    K_integrated_matern = integrated_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # RBF Euclidean kernel
    rbf_euclidean_kernel = gpytorch.kernels.RBFKernel()
    rbf_euclidean_kernel.lengthscale = lengthscale_euclidean
    rbf_euclidean_kernel.to(device)
    K_rbf_euclidean = rbf_euclidean_kernel.forward(x1, x2).cpu().detach().numpy()

    # Euclidean Matérn kernel
    euclidean_matern_kernel = gpytorch.kernels.MaternKernel(nu=smoothness)
    euclidean_matern_kernel.lengthscale = lengthscale_euclidean
    euclidean_matern_kernel.to(device)
    K_euclidean_matern = euclidean_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Distance between x1 and x2
    distance = lorentz_distance_torch(x1, x2)
    distance_np = distance.cpu().detach().numpy()

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distance_np, K_rbf, color='navy', linewidth=1.5)
    plt.plot(distance_np, K_integrated_matern, color='crimson', linewidth=1.5)
    plt.plot(distance_np, K_rbf_euclidean, color='dodgerblue', linewidth=1.5)
    plt.plot(distance_np, K_euclidean_matern, color='gold', linewidth=1.5)
    ax.tick_params(labelsize=22)
    ax.set_xlabel('dH(x1,x2)', fontsize=30)
    ax.set_ylabel('k(x1, x2)', fontsize=30)
    ax.legend(['Hyperbolic RBF', 'Hyperbolic Matérn', 'Euclidean RBF', 'Euclidean Matérn'], fontsize=24)
    plt.show()
