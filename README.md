# PDEsByNNs
This repository contains three Jupyter notebooks illustrating different approaches to solve partial differential equations (PDEs) by means of neural networks (NNs), which are:

## Dependencies
All codes are tested with [TensorFlow](https://www.tensorflow.org/) versions `2.3.0` and `2.4.1`.

## [Physics-informed neural networks (PINNs)](https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb)

<a href="https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We describe the PINN approach in the notebook [`PINN_Solver.ipynb`](https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb) for approximating the solution $u:[0,T] \times \mathcal{D} \to \mathbb{R}$ of an evolution equation

$$
\begin{align}
    \partial_t u (t,x) + \mathcal{N}[u](t,x) &= 0, && (t,x) \in (0,T] \times \mathcal{D},\\
    u(0,x) &= u_0(x) \quad && x \in \mathcal{D},
\end{align}
$$

where $\mathcal{N}$ is a nonlinear differential operator acting on $u$, 
$\mathcal{D} \subset \mathbb{R}^d$ a bounded domain,
$T$ denotes the final time and
$u_0: \mathcal{D} \to \mathbb{R}$ the prescribed initial data.

### Literature
PINNs have been proposed in
- Maziar Raissi, Paris Perdikaris, George Em Karniadakis. *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*. [arXiv 1711.10561](https://arxiv.org/abs/1711.10561) 
- Maziar Raissi, Paris Perdikaris, George Em Karniadakis. *Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations*. [arXiv 1711.10566](https://arxiv.org/abs/1711.10566) 
- Maziar Raissi, Paris Perdikaris, George Em Karniadakis. *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. J. Comp. Phys. 378 pp. 686-707 [DOI: 10.1016/j.jcp.2018.10.045](https://www.sciencedirect.com/science/article/pii/S0021999118307125) 

## [Feynman-Kac solver](https://github.com/janblechschmidt/PDEsByNNs/blob/main/Feynman_Kac_Solver.ipynb)

<a href="https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/Feynman_Kac_Solver.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In the notebook [`Feynman_Kac_Solver.ipynb`](https://github.com/janblechschmidt/PDEsByNNs/blob/main/Feynman_Kac_Solver.ipynb) we illustrate a PDE solver based on the *Feynman-Kac formula*.
We consider the solution by neural network methods of a class of partial differential equations which arise as the *backward Kolmogorov equation*, i.e., linear parabolic second-order PDEs in non-divergence form

$$
\begin{aligned}
    \partial_t u(t,x) + \frac{1}{2} \sigma \sigma^T(t,x) : \nabla^2 u(t,x) + \mu(t,x) \cdot \nabla u(t,x) 
    &= 0, 			\quad && (t,x) \in [0,T) \times \mathbb R^d,\\
    u(T,x) &= g(x), 	\quad && x \in \mathbb R^d.
\end{aligned}
$$

We consider the pure Cauchy problem, allowing the state variable $x$ to vary throughout $\mathbb R^d$, were $d \in \mathbb{N}$ is the spatial dimension, 
$\nabla u(t,x)$ and $\nabla^2 u(t,x)$ denote the gradient and Hessian of the function $u$, respectively, the colon $:$ denotes the Frobenius inner product of $d \times d$ matrices, i.e., $A:B = \sum_{i,j=1}^d a_{ij} \, b_{ij}$, and the dot $\cdot$ the Euclidean inner product on $\mathbb R^d$.
We assume the coefficient functions $\mu\colon[0,T] \times \mathbb R^d \to \mathbb R^d$ (drift) and $\sigma\colon[0,T] \times\mathbb R^d \to \mathbb R^{d \times d}$ (diffusion) to be globally Lipschitz continuous.
Due to the stochastic process connection, the backward Kolmogorov equation is posed as a *final time problem* with data prescribed at time $t=T$ given by a function $g\colon \mathbb R^d \to \mathbb R$.
The simple change of variables $t \mapsto T - t$ leads to the more familiar initial value form

$$
\begin{aligned}
   \partial_t u(t,x) - \frac{1}{2} \sigma \sigma^T(t,x) : \nabla^2 u(t,x) - \mu(t,x) \cdot \nabla u(t,x) 
   &= 0, 			\quad && (t,x) \in (0,T] \times \mathbb R^d,\\
   u(0,x) &= g(x), 	\quad && x \in\mathbb R^d.
\end{aligned}
$$

Equations in non-divergence form like the backward Kolmogorov equation with  leading term $\sigma \sigma^T(t,x) \colon \nabla^2u(t,x)$ typically arise in the context of stochastic differential equations (SDEs) due to the Itô formula.

The goal of the computations that are carried out in this notebook is to approximate the solution of the PDE at a fixed time $t=0$, i.e., the function $u(0,x)$ where $x$ varies over some $d$-dimensional hypercube $\mathcal{D} := [a,b] \subset \mathbb{R}^d$ with endpoint vectors $a \le b$ (component-wise).

### Literature
This solver has been proposed in

- Beck, Christian, et al. *Solving stochastic differential equations and Kolmogorov equations by means of deep learning*. [arXiv 1806.00421](https://arxiv.org/abs/1806.00421).

## Deep BSDE solver


# Paper: Three Ways to Solve Partial Differential Equations with Neural Networks --- A Review

## Abstract
Neural networks are increasingly used to construct numerical solution methods for partial differential equations.
In this expository review, we introduce and contrast three important recent approaches attractive in their simplicity and their suitability for high-dimensional problems: physics-informed neural networks, methods based on the Feynman-Kac formula and the Deep BSDE solver.
The article is accompanied by a suite of expository software in the form of Jupyter notebooks in which each basic methodology is explained step by step, allowing for a quick assimilation and experimentation.
An extensive bibliography summarizes the state of the art.

**Keywords**: partial differential equation; Hamilton-Jacobi-Bellman equations; neural networks, curse of dimensionality, Feynman-Kac, backward differential equation, stochastic process

## arXiv preprint


## Citation
