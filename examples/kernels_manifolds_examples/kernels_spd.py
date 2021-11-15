import numpy as np
import random
import torch
import gpytorch
import matplotlib.pyplot as plt

from BoManifolds.Riemannian_utils.spd_utils import logmap, expmap
from BoManifolds.Riemannian_utils.spd_utils_torch import affine_invariant_distance_torch, \
    symmetric_matrix_to_vector_mandel_torch
from BoManifolds.kernel_utils.kernels_spd import SpdRiemannianGaussianKernel, SpdRiemannianIntegratedMaternKernel, \
    SpdAffineInvariantApproximatedGaussianKernel

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

'''
This example shows the computation of various kernels on the symmetric-positive-definite (SPD) matrix manifold of 
dimension 2. 
The kernels are computed between the starting point of a geodesic and points equally spaced along this geodesic.

The following kernels are considered:
 Geometry-aware kernels: 
  - SpdRiemannianGaussianKernel [1]
  - SpdRiemannianIntegratedMaternKernel [2] (allows the optimization of the smoothness parameter)
  - SpdAffineInvariantApproximatedGaussianKernel [3] (approximation where the distance is replaced by the Riemannian 
    distance)
 Euclidean kernels:
  - RBFKernel
  - MaternKernel

The example displays the kernel value in function of the distance on the manifold for each of the considered kernels.

Note: the kernel parameters (lengthscale and smoothness) may not have the same impact depending on the kernels, i.e., 
Riemannian and Euclidean kernels are not expected to behave exactly similarly for the same parameters values.

References:
[1] P. Sawyer. 
The heat equation on the spaces of positive definite matrices. 
Canadian Journal of Mathematics, 44(3):624–651, 1992.

[2] N. Jaquier, V. Borovitskiy, A. Smolensky, A. Terenin, T. Asfour, and L. Rozo. 
Bayesian Optimization meets Riemannian Manifolds in Robot Learning. 
In Conference on Robot Learning, 2021.

[3] N. Jaquier, L. Rozo, S. Calinon, and M. Bürger. 
Bayesian Optimization meets Riemannian Manifolds in Robot Learning. 
In Conference on Robot Learning, pages 233–246, 2019. 

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
    dim = 2

    # Generate data
    nb_data = 100
    if dim == 2:
        base = np.array([[12., 6.], [6., 17.]])
        point = np.array([[15., -8.], [-8., 12.]])
    else:
        raise NotImplementedError

    u = logmap(point, base)
    data = np.array([expmap(t/nb_data * u, base) for t in range(nb_data)])

    x1_mat = torch.from_numpy(data).to(device)
    x2_mat = torch.from_numpy(point).to(device)
    x1 = symmetric_matrix_to_vector_mandel_torch(x1_mat)
    x2 = symmetric_matrix_to_vector_mandel_torch(x2_mat)[None]

    # Kernels parameters
    lengthscale = 0.5
    smoothness = 2.5

    # RBF kernel
    rbf_kernel = SpdRiemannianGaussianKernel(dim=dim)
    rbf_kernel.lengthscale = np.sqrt(2.)
    rbf_kernel.to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Integrated Matérn kernel
    integrated_matern_kernel = SpdRiemannianIntegratedMaternKernel(dim, nu=smoothness)
    integrated_matern_kernel.lengthscale = np.sqrt(2.)
    integrated_matern_kernel.to(device)
    K_integrated_matern = integrated_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Approximated RBF kernel
    rbf_approx_kernel = SpdAffineInvariantApproximatedGaussianKernel(dim)
    rbf_approx_kernel.beta = 1. / lengthscale ** 2
    rbf_approx_kernel.to(device)
    K_rbf_approx = rbf_approx_kernel.forward(x1, x2).cpu().detach().numpy()

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
    distance = affine_invariant_distance_torch(x1_mat, x2_mat[None])
    distance_np = distance.cpu().detach().numpy()

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distance_np, K_rbf, color='navy', linewidth=1.5)
    plt.plot(distance_np, K_rbf_approx, color='darkviolet', linewidth=1.5)
    plt.plot(distance_np, K_integrated_matern, color='crimson', linewidth=1.5)
    plt.plot(distance_np, K_rbf_euclidean, color='dodgerblue', linewidth=1.5)
    plt.plot(distance_np, K_euclidean_matern, color='gold', linewidth=1.5)
    ax.tick_params(labelsize=22)
    ax.set_xlabel('dS++(x1,x2)', fontsize=30)
    ax.set_ylabel('k(x1,x2)', fontsize=30)
    ax.legend(['SPD RBF', 'SPD Approx. RBF', 'SPD Matérn', 'Euclidean RBF', 'Euclidean Matérn'], fontsize=24)
    plt.show()
