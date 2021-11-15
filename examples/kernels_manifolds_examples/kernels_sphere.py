import numpy as np
import random
import torch
import gpytorch

import matplotlib.pyplot as plt

from BoManifolds.Riemannian_utils.sphere_utils import logmap, expmap
from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch
from BoManifolds.kernel_utils.kernels_sphere import SphereRiemannianGaussianKernel, SphereRiemannianMaternKernel, \
    SphereRiemannianIntegratedMaternKernel, SphereApproximatedGaussianKernel

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

'''
This example shows the computation of various kernels on the sphere manifold. 
The kernels are computed between the starting point of a geodesic and points equally spaced along this geodesic.

The following kernels are considered:
 Geometry-aware kernels: 
  - SphereRiemannianGaussianKernel [2]
  - SphereRiemannianMaternKernel [2]
  - SphereRiemannianIntegratedMaternKernel [3] (allows the optimization of the smoothness parameter)
  - SphereApproximatedGaussianKernel [1] (approximation where the distance is replaced by the Riemannian distance)
 Euclidean kernels:
  - RBFKernel
  - MaternKernel

The example displays the kernel value in function of the distance on the manifold for each of the considered kernels.

Note: the kernel parameters (lengthscale and smoothness) may not have the same impact depending on the kernels, i.e., 
Riemannian and Euclidean kernels are not expected to behave exactly similarly for the same parameters values.

References:
[1] N. Jaquier, L. Rozo, S. Calinon, and M. Bürger. 
Bayesian Optimization meets Riemannian Manifolds in Robot Learning. 
In Conference on Robot Learning, pages 233–246, 2019. 

[2] V. Borovitskiy, A. Terenin, P. Mostowsky, and M. Deisenroth. 
Matérn Gaussian Processes on Riemannian Manifolds. 
In Advances in Neural Information Processing Systems, pages 12426–12437, 2020.

[3] N. Jaquier, V. Borovitskiy, A. Smolensky, A. Terenin, T. Asfour, and L. Rozo. 
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

    # Manifold dimension
    dim_data = 4
    dim_sphere = dim_data - 1  # We are on S^d embedded in R^d+1
    # Kernels parameters
    lengthscale = 0.5
    smoothness = 2.5

    # Generate data
    nb_data = 100
    base = np.zeros((dim_data, 1))
    base[0] = 1.
    point = np.zeros((dim_data, 1))
    point[0] = -.9
    point[1] = .1
    point = point/np.linalg.norm(point)
    u = logmap(point, base)
    data = np.array([expmap(t/nb_data * u, base) for t in range(nb_data)])[:, :, 0]

    x1 = torch.from_numpy(data).to(device)
    x2 = torch.from_numpy(point.T).to(device)

    # RBF kernel
    rbf_kernel = SphereRiemannianGaussianKernel(dim_sphere)
    rbf_kernel.lengthscale = lengthscale
    rbf_kernel.to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Matérn kernel
    matern_kernel = SphereRiemannianMaternKernel(dim=dim_sphere, nu=smoothness)
    matern_kernel.lengthscale = lengthscale
    matern_kernel.to(device)
    K_matern = matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Integrated Matérn kernel
    integrated_matern_kernel = SphereRiemannianIntegratedMaternKernel(dim=dim_sphere, nu=smoothness)
    integrated_matern_kernel.lengthscale = lengthscale
    integrated_matern_kernel.to(device)
    K_integrated_matern = integrated_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Approximated RBF kernel
    rbf_approx_kernel = SphereApproximatedGaussianKernel(dim_sphere)
    rbf_approx_kernel.beta = 1./lengthscale**2
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
    distance = sphere_distance_torch(x1, x2)
    distance_np = distance.cpu().detach().numpy()

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distance_np, K_rbf, color='navy', linewidth=2)
    plt.plot(distance_np, K_rbf_approx, color='darkviolet', linewidth=2)
    plt.plot(distance_np, K_integrated_matern, color='crimson', linewidth=3)
    plt.plot(distance_np, K_integrated_matern, color='teal', linewidth=1)
    plt.plot(distance_np, K_rbf_euclidean, color='dodgerblue', linewidth=2)
    plt.plot(distance_np, K_euclidean_matern, color='gold', linewidth=2)
    ax.tick_params(labelsize=22)
    ax.set_xlabel('dS(x1,x2)', fontsize=30)
    ax.set_ylabel('k(x1,x2)', fontsize=30)
    ax.legend(['Sphere RBF', 'Sphere Approx. RBF', 'Sphere Matérn', 'Sphere Integrated Matérn', 'Euclidean RBF', 'Euclidean Matérn'], fontsize=24)
    plt.show()
