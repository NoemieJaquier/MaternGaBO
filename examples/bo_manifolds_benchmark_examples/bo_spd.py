import os
import types
from argparse import ArgumentParser
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import gpytorch
import gpytorch.kernels
import botorch
import botorch.acquisition

import pymanopt.manifolds as manifolds
import BoManifolds.pymanopt_addons.manifolds as additional_manifolds

import BoManifolds.kernel_utils.kernels_spd as kernel_spd

from BoManifolds.manifold_optimization.manifold_optimize import joint_optimize_manifold
from BoManifolds.euclidean_optimization.euclidean_constrained_optimize import joint_optimize
from BoManifolds.manifold_optimization.robust_trust_regions import TrustRegions
from BoManifolds.manifold_optimization.constrained_trust_regions import StrictConstrainedTrustRegions
import BoManifolds.test_functions_bo.test_functions_manifolds as test_functions_manifolds
from BoManifolds.Riemannian_utils.spd_utils import vector_to_symmetric_matrix_mandel, spd_sample, \
    symmetric_matrix_to_vector_mandel
from BoManifolds.Riemannian_utils.spd_utils_torch import vector_to_symmetric_matrix_mandel_torch, \
    symmetric_matrix_to_vector_mandel_torch

dirname = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print('Device ', device)
else:
    device = 'cpu'

torch.set_default_dtype(torch.float32)

'''
This example shows the use of Bayesian optimization on the symmetric positive-definite (SPD) matrix manifold to optimize
 a benchmark function. 

The type of BO can be chosen by the user via the --bo argument: The Geometry-aware Bayesian optimization (GaBO) takes
 the manifold geometry into account through the kernel definition and acquisition function optimization (see [1], [3]),
 while the classical Euclidean BO (EuBO) ignores the manifold geometry and uses classical kernels with constrained 
 optimization of acquisition functions. Moreover the Cholesky BO (CholBO) applies the classical BO on the Cholesky 
 decomposition of SPD matrices.

The dimension of the manifold can be modified using the --dimension argument. Note that the dimension d corresponds to 
 the dimension of the SPD manifold of d x d matrices.

The kernel function can be chosen by the user via the --kernel argument:
 The following kernels are available for GaBO: 
  - SpdRiemannianGaussianKernel [4]
  - SpdRiemannianIntegratedMaternKernel [3] (allows the optimization of the smoothness parameter)
  - SpdProductOfRSManifoldsRiemannianGaussianKernel [3] (product of R^2 x S^1 or R^3 x S^3 kernels applied on the 
    eigendecomposition of SPD matrices, where the SO part is transformed to S)
  - SpdProductOfRSOManifoldsRiemannianGaussianKernel [3] (product of R^d x SO(d) kernels applied on the 
    eigendecomposition of SPD matrices)
  - SpdAffineInvariantApproximatedGaussianKernel [1] (approximation where the distance is replaced by the Riemannian 
    distance)
 The following kernels are available for EuBO and CholBO (from GPyTorch):
  - RBFKernel
  - MaternKernel
 Additional priors on the kernel parameters can be introduced as arguments.

The acquisition function (Expected improvement) is optimized on the manifold with trust regions on Riemannian manifolds, 
 originally implemented in pymanopt. A robust version is used here to avoid crashing if NaNs or zero values occur during 
 the optimization.

The benchmark function can be chosen among various options, see --benchmark argument options at the end of this file, or 
 the test_function_bo/test_function_manifolds.py file (default: Ackley). 
 The benchmark function, defined on the tangent space of 2*identity, is projected on the sphere with the exponential
 map (i.e. the logarithm map is used to determine the function value). 

The number of BO iterations is set by the user via the --nb_iter_bo argument. 
The initial points for the BO can be modified by changing the seed number (--seed argument).

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the SPD manifold) is printed at the end of the queries. 
The following graphs are produced by this example:
- the convergence graph shows the distance between two consecutive iterations and the best function value found by the
    BO at each iteration. Note that the randomly generated initial data are displayed.

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

[4] P. Sawyer. 
The heat equation on the spaces of positive definite matrices. 
Canadian Journal of Mathematics, 44(3):624–651, 1992.

This file is part of the MaternGaBO library.
Authors: Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo, 2021
License: MIT
Contact: noemie.jaquier@kit.edu
'''

def get_bo_attributes(manifold_name, kernel_name, acquisition_name, test_function_name):
    if hasattr(manifolds, manifold_name):
        manifold = getattr(manifolds, manifold_name)
    elif hasattr(additional_manifolds, manifold_name):
        manifold = getattr(additional_manifolds, manifold_name)
    else:
        raise RuntimeError("No such manifold.")
    
    if hasattr(gpytorch.kernels, kernel_name):
        kernel = getattr(gpytorch.kernels, kernel_name)
    elif hasattr(kernel_spd, kernel_name):
        kernel = getattr(kernel_spd, kernel_name)
    else:
        raise RuntimeError("No such kernel.")

    acquisition = getattr(botorch.acquisition, acquisition_name)

    test_function = getattr(test_functions_manifolds, test_function_name)
    
    return manifold, kernel, acquisition, test_function


def get_bounds(manifold, dimension, bo_type):

    manifold.min_eig = 0.001
    manifold.max_eig = 5.
    dim_vec = int(dimension + dimension * (dimension - 1) / 2)
    if bo_type == "CholBO":
        lower_bound = - np.sqrt(manifold.max_eig) * torch.ones(dim_vec, dtype=torch.float64)
        upper_bound = np.sqrt(manifold.max_eig) * torch.ones(dim_vec, dtype=torch.float64)
    else:
        lower_bound = torch.cat((manifold.min_eig * torch.ones(dimension, dtype=torch.float64),
                                 - manifold.max_eig / np.sqrt(2) * torch.ones(dim_vec - dimension,
                                                                              dtype=torch.float64)))
        upper_bound = torch.cat((manifold.max_eig * torch.ones(dimension, dtype=torch.float64),
                                 manifold.max_eig / np.sqrt(2) * torch.ones(dim_vec - dimension,
                                                                            dtype=torch.float64)))
    bounds = torch.stack([lower_bound, upper_bound])

    return bounds


def update_random_function_manifold(manifold, bounds):
    # The original one samples only eigenvalues between 1 and 2.
    # We need to specify the minimum and maximum eigenvalues of the random matrices (done in bound definition).
    manifold.rand = types.MethodType(spd_sample, manifold)


def get_constraints(bo_type, manifold):
    if bo_type == "GaBO":
        # Define constraints on maximum eigenvalue. The inequality constrains should be defined to be satisfied if >= 0.
        def max_eig_constraint(x):
            eig = torch.linalg.eigvalsh(x)
            return manifold.max_eig - eig.max()

        return [max_eig_constraint]

    elif bo_type == "CholBO":
        # Define constraints on maximum eigenvalue.
        def max_eig_constraint(x_chol):
            dim = manifold._n
            indices = np.tril_indices(dim)
            xL = np.zeros((dim, dim))
            xL[indices] = x_chol

            x_mat = np.dot(xL, xL.T)
            eig = np.linalg.eigvals(x_mat)
            return manifold.max_eig - np.max(eig)

        constraints = [{'type': 'ineq', 'fun': max_eig_constraint}]
        return constraints

    else:
        # Define constraints to be SPD
        def pd_constraint(x):
            x_mat = vector_to_symmetric_matrix_mandel(x)
            eig = np.linalg.eigvals(x_mat)
            # We put here a small value min_eig to be sure that the optimizer satisfies the constraint > 0
            return np.min(eig) - manifold.min_eig

        # Define constraints on the optimization domain (max eigenvalue)
        def max_eig_constraint(x):
            x_mat = vector_to_symmetric_matrix_mandel(x)
            eig = np.linalg.eigvals(x_mat)
            return manifold.max_eig - np.max(eig)

        return [{'type': 'ineq', 'fun': pd_constraint}, {'type': 'ineq', 'fun': max_eig_constraint}]


def get_preprocessing(bo_type):
        if bo_type == "GaBO":
            return vector_to_symmetric_matrix_mandel_torch
        else:
            return None


def get_postprocessing(bo_type, manifold):
        if bo_type == "GaBO":
            return symmetric_matrix_to_vector_mandel_torch

        elif bo_type == "CholBO":
            # Define sampling post processing function to ensure that constraints are satisfied for sampled points
            def post_processing_init(x):
                d_vec = x.shape[-1]
                d_mat = int((-1.0 + (1.0 + 8.0 * d_vec) ** 0.5) / 2.0)

                # Tranform the samples in a Mandel notation form. We can do it by squaring the d first elements and
                # squaring the next ones by keeping the original sign. We can do this operation because we defined
                # the bounds of the  Cholesky decomposition equivalent to sqrt of the corresponding bounds for
                # Mandel notation.
                proc_x = x.clone()
                proc_x = proc_x.view(-1, proc_x.shape[-1])
                proc_x[:, 0:d_mat] = torch.mul(proc_x[:, 0:d_mat], proc_x[:, 0:d_mat])
                proc_x[:, d_mat:] = torch.mul(torch.sign(proc_x[:, d_mat])[:, None],
                                              torch.mul(proc_x[:, d_mat:], proc_x[:, d_mat:]))

                # Transform into a matrix to check the constraints
                x_mat = vector_to_symmetric_matrix_mandel_torch(proc_x)

                # Cholesky vector initialization
                x_chol = torch.zeros(proc_x.shape, dtype=x.dtype)
                indices = np.tril_indices(x_mat.shape[-1])

                for n in range(x_mat.shape[0]):
                    # Check constraints
                    eigdec = torch.eig(x_mat[n], eigenvectors=True)
                    eigvals = eigdec.eigenvalues[:, 0]
                    eigvecs = eigdec.eigenvectors

                    eigvals[eigvals <= 0.] = manifold.min_eig  # PD constraint
                    eigvals[eigvals > manifold.max_eig] = manifold.max_eig  # Max eigenvalue constraint

                    x_mat[n] = torch.mm(torch.mm(eigvecs, torch.diag(eigvals)), torch.inverse(eigvecs))

                    # Cholesky decomposition
                    x_chol[n] = torch.cholesky(x_mat[n])[indices]

                return x_chol.view(x.shape)

            return post_processing_init

        else:
            # Define sampling post processing function to ensure that constraints are satisfied for sampled points
            def post_processing_init(x):
                x_mat = vector_to_symmetric_matrix_mandel_torch(x)
                init_shape = x_mat.shape
                x_mat = x_mat.view(-1, x_mat.shape[-2], x_mat.shape[-1])

                for n in range(x_mat.shape[0]):
                    eigdec = torch.eig(x_mat[n], eigenvectors=True)
                    eigvals = eigdec.eigenvalues[:, 0]
                    eigvecs = eigdec.eigenvectors

                    eigvals[eigvals <= 0.] = manifold.min_eig  # PD constraint
                    eigvals[eigvals > manifold.max_eig] = manifold.max_eig  # Max eigenvalue constraint

                    x_mat[n] = torch.mm(torch.mm(eigvecs, torch.diag(eigvals)), torch.inverse(eigvecs))

                x_mat = x_mat.view(init_shape)
                return symmetric_matrix_to_vector_mandel_torch(x_mat)
            return post_processing_init


def main(manifold_name, dimension, kernel_name, acquisition_name, bo_type, test_function_name, nb_iter_bo, seed_id,
         nu, nu_prior_params, lengthscale_prior_params):
    np.random.seed(1234)

    # Parameters
    nb_seeds = 100
    nb_data_init = 5

    # Recover the manifold, kernel and acquisition functions types
    manifold_type, kernel_type, acquisition_type, test_function_type = get_bo_attributes(manifold_name, kernel_name,
                                                                                         acquisition_name,
                                                                                         test_function_name)

    # Generate sequence of random seeds
    seeds = np.random.randint(0, 2 ** 16, size=nb_seeds)

    # Set numpy and pytorch seeds
    random.seed(seeds[seed_id])
    np.random.seed(seeds[seed_id])
    torch.manual_seed(seeds[seed_id])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Execution of BO for one seed
    print('Seed ' + str(seed_id))

    # Instantiate the manifold
    manifold = manifold_type(dimension)

    # Test function and optimum
    test_function_class = test_function_type(manifold)
    if bo_type == "CholBO":
        test_function_class.cholesky = True
    test_function = test_function_class.compute_function_torch
    true_min, true_opt_val = test_function_class.optimum()

    # Specify the optimization domain
    bounds = get_bounds(manifold, dimension, bo_type)

    # If needed, update the random function of the manifold (Euclidean, and SPD)
    update_random_function_manifold(manifold, bounds.detach().numpy())

    # Define the pre/post-processing functions in function of the BO type and of the manifold
    preprocessing_fct = get_preprocessing(bo_type)  # Vector to matrix for GaBO optimization of the acquisition function
    postprocessing_fct = get_postprocessing(bo_type, manifold)  # Matrix to vector after GaBO optimization of the
    # acquisition function; Ensure SPD data / Cholesky decompositions for EuBO / CholBO initializations;

    # Generate random data on the manifold
    x_init = np.array([manifold.rand() for n in range(nb_data_init)])
    if isinstance(manifold, manifolds.SymmetricPositiveDefinite):
        if bo_type == "CholBO":
            x_init = np.array([np.linalg.cholesky(x_init[i])[np.tril_indices(dimension)] for i in range(nb_data_init)])
        else:
            x_init = np.array([symmetric_matrix_to_vector_mandel(x_init[i]) for i in range(nb_data_init)])

    x_data = torch.tensor(x_init)

    # Function value for initial data
    y_data = torch.zeros(nb_data_init, dtype=torch.float64)
    for n in range(nb_data_init):
        y_data[n] = test_function(x_data[n])

    # Define the kernels prior parameters
    if nu_prior_params is None:
        nu_prior = gpytorch.priors.torch_priors.GammaPrior(2.0, 0.1)
    else:
        nu_prior = gpytorch.priors.torch_priors.GammaPrior(nu_prior_params[0], nu_prior_params[1])
    if lengthscale_prior_params is None:
        lengthscale_prior = None
        if bo_type == "CholBO":
            lengthscale_prior = gpytorch.priors.torch_priors.GammaPrior(3.0, 6.0)
    else:
        lengthscale_prior = gpytorch.priors.torch_priors.UniformPrior(lengthscale_prior_params[0],
                                                                      lengthscale_prior_params[1]),

    # Define the base kernel
    base_kernel = kernel_type(dim=dimension, nu=nu, nu_prior=nu_prior, lengthscale_prior=lengthscale_prior)
    base_kernel.to(device)

    if "SpdProductOf" in kernel_name and isinstance(manifold, manifolds.SymmetricPositiveDefinite):
        base_kernel.spd_input = True

    # Define the kernel function
    k_fct = gpytorch.kernels.ScaleKernel(base_kernel,
                                         outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    k_fct.to(device)

    # Define the GPR model
    # A constant mean function is already included in the model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                          noise_constraint=
                                                                          gpytorch.constraints.GreaterThan(1e-8),
                                                                          initial_value=noise_prior_mode)
    lik_fct.to(device)
    model = botorch.models.SingleTaskGP(x_data, y_data[:, None], covar_module=k_fct, likelihood=lik_fct)
    model.to(device)

    # Define the marginal log-likelihood
    mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    mll_fct.to(device)

    # Define the constraints and processing functions in function of the BO type and of the manifold
    constraints = get_constraints(bo_type, manifold)

    if bo_type == "GaBO":
        # Define the solver on the manifold
        if constraints is None:
            solver = TrustRegions(maxiter=200)
        else:
            solver = StrictConstrainedTrustRegions(mingradnorm=1e-3, maxiter=50)
        # Do we approximate the Hessian
        approx_hessian = True

    num_restarts = 5

    # Initialize best observation and function value list
    new_best_f, index = y_data.min(0)
    best_x = [x_data[index]]
    best_f = [new_best_f]

    # BO loop
    x_data = x_data.to(device)
    y_data = y_data.to(device)
    for iteration in range(nb_iter_bo):
        # Sample kernel parameters
        if 'Matern' in kernel_name and nu is None:
            botorch.optim.utils.sample_all_priors(mll_fct.model.covar_module)
        # Fit GP model
        botorch.fit_gpytorch_model(mll=mll_fct)

        # Define the acquisition function
        acq_fct = acquisition_type(model=model, best_f=best_f[-1], maximize=False)
        acq_fct.to(device)

        # Get new candidate
        if bo_type == "GaBO":
            new_x = joint_optimize_manifold(acq_fct, manifold, solver, q=1, num_restarts=num_restarts, raw_samples=100,
                                            bounds=bounds,
                                            pre_processing_manifold=preprocessing_fct,
                                            post_processing_manifold=postprocessing_fct,
                                            approx_hessian=approx_hessian, inequality_constraints=constraints)
        else:
            new_x = joint_optimize(acq_fct, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=100,
                                   constraints=constraints, post_processing_init=postprocessing_fct)

        # Get new observation
        new_y = test_function(new_x)[0].to(device)

        # Update training points
        x_data = torch.cat((x_data, new_x))
        y_data = torch.cat((y_data, new_y))

        # Update best observation
        new_best_f, index = y_data.min(0)
        best_x.append(x_data[index])
        best_f.append(new_best_f)

        # Update the model
        model.set_train_data(x_data, y_data, strict=False)  # strict False necessary to add datapoints

        print("Iteration " + str(iteration) + "\t Best f " + str(new_best_f.item()))

    # To numpy
    x_eval = x_data.cpu().numpy()
    y_eval = y_data.cpu().numpy()[:, None]

    best_x_np = np.array([x.cpu().detach().numpy() for x in best_x])
    best_f_np = np.array([f.cpu().detach().numpy() for f in best_f])[:, None]

    # Compute distances between consecutive x's and best evaluation for each iteration
    neval = x_eval.shape[0]
    distances = np.zeros(neval - 1)
    for n in range(neval - 1):
        distances[n] = manifold.dist(vector_to_symmetric_matrix_mandel(x_eval[n + 1, :]),
                                     vector_to_symmetric_matrix_mandel(x_eval[n, :]))

    Y_best = np.ones(neval)
    for i in range(neval):
        Y_best[i] = y_eval[:(i + 1)].min()

    #  Plot distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(neval - 1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    plt.grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(np.array(range(neval)), Y_best, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--bo", dest="bo", default="GaBO",
                        help="Set the BO type. Options: GaBO, EuBO, CholBO")
    parser.add_argument("--dimension", dest="dimension", type=int, default=2,
                        help="Set the dimension of the manifold")
    parser.add_argument("--benchmark", dest="benchmark_function", default="Ackley",
                        help="Set the benchmark function. Options: Ackley, Rosenbrock, StyblinskiTang, Levy, "
                             "ProductOfSines, ...")
    parser.add_argument("--kernel", dest="kernel", default="SpdRiemannianGaussianKernel",
                        help="Set the kernel. Options: RBFKernel, MaternKernel "
                             "SpdRiemannianGaussianKernel, SpdRiemannianIntegratedMaternKernel, "
                             "SpdProductOfRSManifoldsRiemannianGaussianKernel, "
                             "SpdProductOfRSOManifoldsRiemannianGaussianKernel, "
                             "SpdAffineInvariantApproximatedGaussianKernel")
    parser.add_argument("--acquisition", dest="acquisition", default="ExpectedImprovement",
                        help="Set the acquisition function. Options: ExpectedImprovement, ProbabilityOfImprovement, "
                             "UpperConfidenceBound, IntegratedExpectedImprovement, ...")
    parser.add_argument("--seed", dest="seed", type=int, default=0,
                        help="Set the seed ID")
    parser.add_argument("--nb_iter_bo", dest="nb_iter_bo", type=int, default=25,
                        help="Set the number of BO iterations")
    parser.add_argument('--nu', dest="nu", type=float, default=2.5,
                        help="Kernel smoothness parameter, default 2.5 (nu is optimize if given as None for integrated "
                             "Matérn kernels).")
    parser.add_argument('--nu_prior_params', dest="nu_prior_params", nargs='+', default=None,
                        help="Kernel smoothness gamma prior function's parameters (2 values), default None.")
    parser.add_argument('--lengthscale_prior_params', dest="lengthscale_prior_params", nargs='+', default=None,
                        help="Kernel lengthscale uniform prior function's parameters (2 values), default None.")

    args = parser.parse_args()
    
    # BO type
    bo_type = args.bo

    # Manifold
    manifold_name = "SymmetricPositiveDefinite"

    # dimension
    dimension = args.dimension

    # Benchmark test function
    test_function_name = args.benchmark_function

    # Kernel
    kernel_name = args.kernel

    # Acquisition function
    acquisition_name = args.acquisition

    # Seed
    seed_id = args.seed

    # Number of iterations
    nb_iter_bo = args.nb_iter_bo

    # Kernel parameters and priors
    nu = args.nu
    nu_prior_params = args.nu_prior_params
    lengthscale_prior_params = args.lengthscale_prior_params

    # Run the BO
    main(manifold_name, dimension, kernel_name, acquisition_name, bo_type, test_function_name, nb_iter_bo, seed_id,
         nu, nu_prior_params, lengthscale_prior_params)
