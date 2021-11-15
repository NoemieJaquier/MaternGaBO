import numpy as np
import random
import torch
import gpytorch
import matplotlib.pyplot as plt

from BoManifolds.Riemannian_utils.sphere_utils import logmap, expmap
from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch
from BoManifolds.kernel_utils.kernels_torus import TorusProductOfManifoldsRiemannianMaternKernel, \
    TorusProductOfManifoldsRiemannianGaussianKernel

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

'''
This example shows the computation of various kernels on the torus manifold. 
The kernels are computed between the starting point of a geodesic and points equally spaced along this geodesic.

The following kernels are considered:
 Geometry-aware kernels (seen as product of kernels on the circle): 
  - TorusProductOfManifoldsRiemannianGaussianKernel [2]
  - TorusProductOfManifoldsRiemannianMaternKernel [1], [2]
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

[2] V. Borovitskiy, A. Terenin, P. Mostowsky, and M. Deisenroth. 
Matérn Gaussian Processes on Riemannian Manifolds. 
In Advances in Neural Information Processing Systems, pages 12426–12437, 2020.

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
    dim_torus = 3
    dim_data = 2 * dim_torus

    # Generate data
    nb_data = 100
    base = np.zeros((2, 1))
    base[0] = 1.
    data = []
    data_angles = []
    for i in range(dim_torus):
        point = np.random.randn(2)
        point = point / np.linalg.norm(point)
        u = logmap(point, base)
        data.append(np.array([expmap(t / nb_data * u, base) for t in range(nb_data)])[:, :, 0])
        data_angles.append(np.arccos(data[i][:, 0])[:, None])

    x1 = torch.from_numpy(np.hstack(data)).to(device)
    x2 = torch.from_numpy(np.vstack([base for i in range(dim_torus)]).T).to(device)

    # x1_angles = torch.from_numpy(np.hstack(data_angles))
    # x1_angles = x1_angles.to(device)
    # x2_angles = torch.from_numpy(np.zeros((1, dim_torus)))
    # x2_angles = x2_angles.to(device)

    # Kernels parameters
    lengthscale = 0.1
    smoothness = 2.5

    # RBF kernel
    rbf_kernel = TorusProductOfManifoldsRiemannianGaussianKernel(dim_torus)
    for i in range(dim_torus):
        rbf_kernel.torus_kernel.kernels[i].lengthscale = lengthscale
        rbf_kernel.torus_kernel.kernels[i].to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Integrated Matérn kernel
    integrated_matern_kernel = TorusProductOfManifoldsRiemannianMaternKernel(dim=dim_torus, nu=smoothness)
    for i in range(dim_torus):
        integrated_matern_kernel.torus_kernel.kernels[i].lengthscale = lengthscale
        integrated_matern_kernel.torus_kernel.kernels[i].to(device)
    K_integrated_matern = integrated_matern_kernel.forward(x1, x2).cpu().detach().numpy()

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
    squared_distances_list = [sphere_distance_torch(x1[:, 2*i:2*i+2], x2[:, 2*i:2*i+2])**2 for i in range(dim_torus)]
    distance = torch.sqrt(torch.sum(torch.stack(squared_distances_list), dim=0))
    distance_np = distance.cpu().detach().numpy()

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distance_np, K_rbf, color='navy', linewidth=1.5)
    plt.plot(distance_np, K_integrated_matern, color='crimson', linewidth=1.5)
    plt.plot(distance_np, K_rbf_euclidean, color='dodgerblue', linewidth=1.5)
    plt.plot(distance_np, K_euclidean_matern, color='gold', linewidth=1.5)
    ax.tick_params(labelsize=22)
    ax.set_xlabel('dT(x1,x2)', fontsize=30)
    ax.set_ylabel('k(x1,x2)', fontsize=30)
    ax.legend(['Torus RBF', 'Torus Matérn', 'Euclidean RBF', 'Euclidean Matérn'], fontsize=24)
    plt.show()
