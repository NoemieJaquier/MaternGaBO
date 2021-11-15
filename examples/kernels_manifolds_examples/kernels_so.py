import numpy as np
import random
import torch
import gpytorch
from pymanopt.manifolds import SpecialOrthogonalGroup
import matplotlib.pyplot as plt

from BoManifolds.kernel_utils.kernels_so import SORiemannianGaussianKernel, SORiemannianMaternKernel

# Use double precision
torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

'''
This example shows the computation of various kernels on the special orthogonal group. 
The kernels are computed between the starting point of a geodesic and points equally spaced along this geodesic.

The following kernels are considered:
 Geometry-aware kernels: 
  - SORiemannianGaussianKernel [1]
  - SORiemannianMaternKernel [1]
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
    dim_so = 3

    # Define the manifold
    manifold = SpecialOrthogonalGroup(dim_so)

    # Generate data
    nb_data = 100
    base = np.eye(dim_so)
    point = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0.], [0.48, 0.64, 0.6]])
    u = manifold.log(base, point)
    data = np.array([manifold.exp(base, u * t/(nb_data-1)) for t in range(nb_data)])
    x1_mat = data
    x2_mat = point
    x1 = torch.from_numpy(data).to(device).view(-1, dim_so**2)
    x2 = torch.from_numpy(point).to(device).view(-1, dim_so**2)

    # Kernels parameters
    lengthscale = 0.5
    smoothness = 2.5

    # RBF kernel
    rbf_kernel = SORiemannianGaussianKernel(dim_so)
    rbf_kernel.lengthscale = lengthscale
    rbf_kernel.to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Matérn kernel
    matern_kernel = SORiemannianMaternKernel(dim=dim_so, nu=smoothness)
    matern_kernel.lengthscale = lengthscale
    matern_kernel.to(device)
    K_matern = matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # RBF Euclidean kernel
    rbf_euclidean_kernel = gpytorch.kernels.RBFKernel()
    rbf_euclidean_kernel.lengthscale = lengthscale
    rbf_euclidean_kernel.to(device)
    K_rbf_euclidean = rbf_euclidean_kernel.forward(x1, x2).cpu().detach().numpy()

    # Euclidean Matérn kernel
    euclidean_matern_kernel = gpytorch.kernels.MaternKernel(nu=smoothness)
    euclidean_matern_kernel.lengthscale = lengthscale
    euclidean_matern_kernel.to(device)
    K_euclidean_matern = euclidean_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Distance between x1 and x2
    distance_np = np.array([manifold.dist(x1_mat[i], x2_mat) for i in range(nb_data)])

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distance_np, K_rbf, color='navy', linewidth=1.5)
    plt.plot(distance_np, K_matern, color='crimson', linewidth=1.5)
    plt.plot(distance_np, K_rbf_euclidean, color='dodgerblue', linewidth=1.5)
    plt.plot(distance_np, K_euclidean_matern, color='gold', linewidth=1.5)
    ax.tick_params(labelsize=22)
    ax.set_xlabel('dSO(x1,x2)', fontsize=30)
    ax.set_ylabel('k(x1, x2)', fontsize=30)
    ax.legend(['SO RBF', 'SO Matérn', 'Euclidean RBF', 'Euclidean Matérn'], fontsize=24)
    plt.show()
