import os
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
import BoManifolds.kernel_utils.kernels_hyperbolic as kernel_hyperbolic
from BoManifolds.manifold_optimization.manifold_optimize import joint_optimize_manifold
from BoManifolds.euclidean_optimization.euclidean_constrained_optimize import joint_optimize
from BoManifolds.manifold_optimization.robust_trust_regions import TrustRegions
from BoManifolds.manifold_optimization.constrained_trust_regions import StrictConstrainedTrustRegions
import BoManifolds.test_functions_bo.test_functions_manifolds as test_functions_manifolds


if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print('Device ', device)
else:
    device = 'cpu'

torch.set_default_dtype(torch.float32)

dirname = os.path.dirname(os.path.realpath(__file__))

'''
This example shows the use of Bayesian optimization on the hyperbolic Lorentz manifold to optimize a benchmark function. 

The type of BO can be chosen by the user via the --bo argument: The Geometry-aware Bayesian optimization (GaBO) takes
 the manifold geometry into account through the kernel definition and acquisition function optimization (see [1], [3]),
 while the classical Euclidean BO (EuBO) ignores the manifold geometry and uses classical kernels with constrained 
 optimization of acquisition functions.

The dimension of the manifold can be modified using the --dimension argument. Note that the dimension d corresponds to 
 the dimension of the hyperbolic space H^d such that -x(0)^2 + x(1)^2 + ... + x(d)^2 = -1.

The kernel function can be chosen by the user via the --kernel argument:
 The following kernels are available for GaBO: 
  - HyperbolicRiemannianMillsonGaussianKernel [4]
  - HyperbolicRiemannianMillsonIntegratedMaternKernel [3] (allows the optimization of the smoothness parameter)
 The following kernels are available for EuBO (from GPyTorch):
  - RBFKernel
  - MaternKernel
 Additional priors on the kernel parameters can be introduced as arguments.

The acquisition function (Expected improvement) is optimized on the manifold with trust regions on Riemannian manifolds, 
 originally implemented in pymanopt. A robust version is used here to avoid crashing if NaNs or zero values occur during 
 the optimization.

The benchmark function can be chosen among various options, see --benchmark argument options at the end of this file, or 
 the test_function_bo/test_function_manifolds.py file (default: Ackley). 
 The benchmark function, defined on the tangent space of (1, 0, 0, ...), is projected on the sphere with the exponential
 map (i.e. the logarithm map is used to determine the function value). 

The number of BO iterations is set by the user via the --nb_iter_bo argument. 
The initial points for the BO can be modified by changing the seed number (--seed argument).

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the hyperbolic manifold) is printed at the end of the queries. 
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

[4] A. Grigoryan and M. Noguchi. 
The heat kernel on hyperbolic space. 
Bulletin of the London Mathematical Society, 30(6):643–650, 1998.

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
    elif hasattr(kernel_hyperbolic, kernel_name):
        kernel = getattr(kernel_hyperbolic, kernel_name)
    else:
        raise RuntimeError("No such kernel.")

    acquisition = getattr(botorch.acquisition, acquisition_name)

    test_function = getattr(test_functions_manifolds, test_function_name)

    return manifold, kernel, acquisition, test_function


def get_bounds(dimension):
    bounds = torch.stack([- 10. * torch.ones(dimension + 1, dtype=torch.float64),
                          10. * torch.ones(dimension + 1, dtype=torch.float64)])
    return bounds


def get_constraints(bo_type):
    if bo_type == "GaBO":
        return None

    else:
        # Define constraints to be on the hyperbolic manifold
        def minkowski_norm_constraint(x):
            return np.sum(x[1:] ** 2) - x[0] ** 2 + 1.

        return [{'type': 'eq', 'fun': minkowski_norm_constraint}]


def get_preprocessing():
    return None


def get_postprocessing(bo_type):
    if bo_type == "GaBO":
        return None

    else:
        # Define sampling post processing function
        def post_processing_init(x):
            x0 = torch.sqrt(1. + torch.norm(x[..., 1:], dim=[-1]) ** 2)
            x[..., 0] = x0
            return x

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
    test_function = test_function_class.compute_function_torch
    true_min, true_opt_val = test_function_class.optimum()

    # Specify the optimization domain
    bounds = get_bounds(dimension)

    # Define the pre/post-processing functions in function of the BO type and of the manifold
    preprocessing_fct = get_preprocessing()  # None for the hyperbolic manifold
    postprocessing_fct = get_postprocessing(bo_type)  # Ensure data on the hyperbolic manifold for EuBO initializations

    # Generate random data on the manifold
    x_init = np.array([manifold.rand() for n in range(nb_data_init)])
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
        lengthscale_prior = gpytorch.priors.torch_priors.GammaPrior(3.0, 6.0)
    else:
        lengthscale_prior = gpytorch.priors.torch_priors.UniformPrior(lengthscale_prior_params[0],
                                                                      lengthscale_prior_params[1]),

    # Define the base kernel
    base_kernel = kernel_type(dim=dimension, nu=nu, nu_prior=nu_prior, lengthscale_prior=lengthscale_prior)
    base_kernel.to(device)

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
    constraints = get_constraints(bo_type)

    if bo_type == "GaBO":
        # Define the solver on the manifold
        if constraints is None:
            solver = TrustRegions(maxiter=200)
        else:
            solver = StrictConstrainedTrustRegions(mingradnorm=1e-3, maxiter=50)

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
                                            approx_hessian=False, inequality_constraints=constraints)
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
        distances[n] = manifold.dist(x_eval[n + 1, :], x_eval[n, :])

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
    parser.add_argument("--dimension", dest="dimension", type=int, default=3,
                        help="Set the dimension of the manifold")
    parser.add_argument("--benchmark", dest="benchmark_function", default="Ackley",
                        help="Set the benchmark function. Options: Ackley, Rosenbrock, StyblinskiTang, Levy, "
                             "ProductOfSines, ...")
    parser.add_argument("--kernel", dest="kernel", default="HyperbolicRiemannianMillsonGaussianKernel",
                        help="Set the kernel. Options: RBFKernel, MaternKernel "
                             "HyperbolicRiemannianMillsonGaussianKernel, "
                             "HyperbolicRiemannianMillsonIntegratedMaternKernel")
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
    manifold_name = "HyperbolicLorentz"

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

