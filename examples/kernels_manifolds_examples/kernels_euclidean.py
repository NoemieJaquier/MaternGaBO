import numpy as np
import random
import torch
import gpytorch
import matplotlib.pyplot as plt

from BoManifolds.kernel_utils.kernels_euclidean import EuclideanIntegratedMaternKernel

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

if torch.cuda.is_available():
    device = torch.cuda.current_device()
else:
    device = 'cpu'

'''
This example shows the computation of various kernels on the Euclidean space. 
The kernels are computed between the starting point of a geodesic and points equally spaced along this geodesic 
 (= straight line in Euclidean space).

The following kernels are considered:
 Euclidean kernels:
  - RBFKernel
  - MaternKernel
  - EuclideanIntegratedMaternKernel (allows the optimization of the smoothness parameter)

The example displays the kernel value in function of the Euclidean distance for each of the considered kernels.

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
    dim = 4

    # Generate data
    nb_data = 100
    x1 = torch.zeros((nb_data, dim)).to(device)
    x1[:, 0] = torch.linspace(0, 6, nb_data)
    x2 = torch.zeros((1, dim)).to(device)

    # Kernels parameters
    lengthscale = 1.
    smoothness = 2.5

    # RBF kernel
    rbf_kernel = gpytorch.kernels.RBFKernel()
    rbf_kernel.lengthscale = lengthscale
    rbf_kernel.to(device)
    K_rbf = rbf_kernel.forward(x1, x2).cpu().detach().numpy()

    # Matérn kernel
    matern_kernel = gpytorch.kernels.MaternKernel(nu=smoothness)
    matern_kernel.lengthscale = lengthscale
    matern_kernel.to(device)
    K_matern = matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Integrated Matérn kernel
    integrated_matern_kernel = EuclideanIntegratedMaternKernel(dim=dim, nu=smoothness)
    integrated_matern_kernel.lengthscale = lengthscale
    integrated_matern_kernel.to(device)
    K_integrated_matern = integrated_matern_kernel.forward(x1, x2).cpu().detach().numpy()

    # Distance between x1 and x2
    distance = matern_kernel.covar_dist(x1, x2, square_dist=False)
    distance_np = distance.cpu().detach().numpy()

    # Plot kernel value in function of the distance
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(distance_np, K_rbf, color='navy', linewidth=2)
    plt.plot(distance_np, K_matern, color='crimson', linewidth=3)
    plt.plot(distance_np, K_integrated_matern, color='dodgerblue', linewidth=1)
    ax.tick_params(labelsize=22)
    ax.set_xlabel('d(x1,x2)', fontsize=30)
    ax.set_ylabel('k(x1,x2)', fontsize=30)
    ax.legend(['RBF', 'Matern', 'Integrated Matern'], fontsize=24)
    plt.show()


