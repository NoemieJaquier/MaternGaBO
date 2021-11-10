# BoManifolds

This repository contains the source code  to perform Geometry-aware Bayesian Optimization with Riemannian Matérn kernels.

# Dependencies
This code runs with Python>=3.6. It requires the following packages:
- numpy
- scipy
- matplotlib
- pymanopt
- torch
- gpytorch
- botorch
- sympy

# Installation 
To install it, first clone the repository and install the related packages, as explained below.

```
pip install -r requirements.txt
```

# Examples
The following example are available:
- bo_manifold_sphere.py
- bo_manifold_spd.py
- bo_manifold_hyperbolic.py
- bo_manifold_so.py
- bo_manifold_torus.py

For each example, the type of BO, the type of kernel, the dimension of the manifold, and the benchmark function can be chosen.


# References
If you found GaBOtorch useful, we would be grateful if you cite the following reference:

[ [1](https://openreview.net/forum?id=ovRdr3FOIIm) ] N. Jaquier*, V. Borovitskiy*, A. Smolensky, A. Terenin, T. Asfour, and L. Rozo (2021). Geometry-aware Bayesian Optimization in Robotics using Riemannian Matérn Kernels. In Conference on Robot Learning (CoRL).

You can find the video accompanying the paper [here](https://www.youtube.com/watch?v=6awfFRqP7wA&feature=youtu.be).
```
@inproceedings{Jaquier21MaternGaBO,
	author="Jaquier, N. and Borovitskiy, V. and Smolensky, A. and Terenin, A. and Asfour, T. and Rozo, L.", 
	title="Geometry-aware Bayesian Optimization in Robotics using Riemannian Mat\'ern Kernels",
	booktitle="Conference on Robot Learning (CoRL)",
	year="2021",
	pages=""
}
```