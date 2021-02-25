# PDEsByNNs

This repository contains three Jupyter notebooks illustrating different approaches to solve partial differential equations (PDEs) by means of neural networks (NNs).

The notebooks serve as supplementary material to the paper:

## Three Ways to Solve Partial Differential Equations with Neural Networks - A Review

**Abstract**: Neural networks are increasingly used to construct numerical solution methods for partial differential equations.
In this expository review, we introduce and contrast three important recent approaches attractive in their simplicity and their suitability for high-dimensional problems: physics-informed neural networks, methods based on the Feynman-Kac formula and the Deep BSDE solver.
The article is accompanied by a suite of expository software in the form of Jupyter notebooks in which each basic methodology is explained step by step, allowing for a quick assimilation and experimentation.
An extensive bibliography summarizes the state of the art.

**Keywords**: partial differential equation; Hamilton-Jacobi-Bellman equations; neural networks, curse of dimensionality, Feynman-Kac, backward differential equation, stochastic process

**arXiv preprint**: [https://arxiv.org/abs/2102.11802](https://arxiv.org/abs/2102.11802)

**Citation**:

    @misc{blechschmidt2021ways,
      title={Three Ways to Solve Partial Differential Equations with Neural Networks --- A Review}, 
      author={Jan Blechschmidt and Oliver G. Ernst},
      year={2021},
      eprint={2102.11802},
      archivePrefix={arXiv},
      primaryClass={math.NA}
    }

**Dependencies**: All codes are tested with [TensorFlow](https://www.tensorflow.org/) versions `2.3.0` and `2.4.1`.

## [Physics-informed neural networks (PINNs)](https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb)

<a href="https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<br>

We describe the PINN approach in the notebook [`PINN_Solver.ipynb`](https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb) for approximating the solution of a **nonlinear evolution equation on a bounded domain** by a neural network.

### Literature
PINNs have been proposed in
- Maziar Raissi, Paris Perdikaris, George Em Karniadakis. *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*. [arXiv 1711.10561](https://arxiv.org/abs/1711.10561) 
- Maziar Raissi, Paris Perdikaris, George Em Karniadakis. *Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations*. [arXiv 1711.10566](https://arxiv.org/abs/1711.10566) 
- Maziar Raissi, Paris Perdikaris, George Em Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. J. Comp. Phys. 378 pp. 686-707 [DOI: 10.1016/j.jcp.2018.10.045](https://www.sciencedirect.com/science/article/pii/S0021999118307125) 

## [Feynman-Kac solver](https://github.com/janblechschmidt/PDEsByNNs/blob/main/Feynman_Kac_Solver.ipynb)

<a href="https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/Feynman_Kac_Solver.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<br>

In the notebook [`Feynman_Kac_Solver.ipynb`](https://github.com/janblechschmidt/PDEsByNNs/blob/main/Feynman_Kac_Solver.ipynb) we illustrate a PDE solver based on the *Feynman-Kac formula*.
We consider the solution by neural network methods of a class of partial differential equations which arise as the *backward Kolmogorov equation*, i.e., **linear parabolic second-order PDEs in non-divergence form on an unbounded domain in high spatial dimensions**.
The goal of the computations that are carried out in this notebook is to approximate the solution of the PDE at a fixed time where the spatial variable varies over some d-dimensional hypercube.

### Literature
This solver has been proposed in

- Beck, Christian, et al. *Solving stochastic differential equations and Kolmogorov equations by means of deep learning*. [arXiv 1806.00421](https://arxiv.org/abs/1806.00421).

## [Deep BSDE solver](https://github.com/janblechschmidt/PDEsByNNs/blob/main/DeepBSDE_Solver.ipynb)

<a href="https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/DeepBSDE_Solver.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<br>

In this section we extend the methodology of the [Feynman-Kac solver (GitHub)](https://github.com/janblechschmidt/PDEsByNNs/blob/main/Feynman-Kac_Solver.ipynb) to solving *semilinear* PDEs where the reaction term contains lower order terms can depend in a general way on the independent variables as well as on the PDE solution and its gradient.

The implementation addresses the problem of evaluating the PDE solution at a fixed point in time and space.
However, the code can be modified to obtain the solution of the PDE at a fixed time in a domain of interest, as described in the [Feynman-Kac solver (GitHub)](https://github.com/janblechschmidt/PDEsByNNs/blob/main/Feynman-Kac_Solver.ipynb).

### Literature
The Deep BSDE solver has been introduced in

- W. E, J. Han and A. Jentzen. *Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations*. Communications in Mathematics and Statistics, 5, 349–380 (2017), see [published version](https://doi.org/10.1007/s40304-017-0117-6) or [arXiv preprint](https://arxiv.org/abs/1706.04702)
- J. Han, A. Jentzen and W. E. *Solving high-dimensional partial differential equations using deep learning*. PNAS August 21, 2018 115 (34) 8505-8510, see [published version](https://doi.org/10.1073/pnas.1718942115) or [arXiv preprint](https://arxiv.org/abs/1707.02568).
