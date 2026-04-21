# Solving Infinite-Dimensionally Coupled PDEs for Spherical Polymer Diffusion Using Deep Learning

**Authors:** Yi He (heyi2023@lzu.edu.cn), Heng Wang (220220934161@lzu.edu.cn), and Weihua Deng (dengwh@lzu.edu.cn)

**Address:** School of Mathematics and Statistics, Gansu Key Laboratory of Applied Mathematics and Complex Systems, Lanzhou University, Lanzhou 730000, China.

## Environment configuration
```bash
conda env create -f environment.yml
```

## Introduction

This repository provides a **Spherical Deep BSDE solver** for infinite-dimensionally coupled PDEs on spherical manifolds. 

**Core Innovation**: Standard spherical coordinates suffer from severe polar singularities (e.g., terms like $1/\sin\theta \to \infty$). To ensure numerical stability and satisfy the Lipschitz conditions required for BSDE convergence, we introduce a dynamic coordinate rotation scheme. This scheme dynamically maps local spatial updates to the equator, completely eliminating geometric singularities.

The unified coupled PDE system solved in this repository is compactly formulated as:

$$
\frac{\partial}{\partial t} u(\mathbf{x}, n, t) + \mathcal{L}_{\text{diff}} u(\mathbf{x}, n, t) + \mathcal{L}_{\text{jump}} u(\mathbf{x}, n, t) + f\left(\mathbf{x}, n, t, u, \boldsymbol{\sigma}^\top \nabla_{\mathbf{x}} u\right) = 0
$$

where $\mathbf{x} \in \mathbb{S}^2$ is the spatial coordinate, $n$ is the polymer state, $\mathcal{L}_{\text{diff}}$ is the spatial diffusion operator on the sphere, and $\mathcal{L}_{\text{jump}}$ governs the discrete Markov transitions of the polymer. And the terminal condition $u(n,x,T)=g(n,x)$.


## Example
We provide invocation examples corresponding to the equations in our paper within the Python scripts `./exampleR.py` and `./exampleC.py`:
* **exampleR.py**： Demonstrates the solution for the **Fokker-Planck equations** using Real numbers.
* **exampleC.py**： Demonstrates the solution for the **Feynman-Kac equations** using Complex numbers.

You can directly run these scripts to test the models, or freely define and modify any of the components and parameters within the samples for your own experiments.


## More: Interactive Demo
We have deployed an **Interactive Gradient Fitting Demo** on Hugging Face Spaces. 

[Visit the  Demo here](https://huggingface.co/spaces/hy-0003/BSDES)


## Thanks
Our methodology is mainly established upon two foundational works:

**Algorithm Framework**: We extend the Deep BSDE approach introduced in **"Solving bivariate kinetic equations for polymer diffusion using deep learning"** to handle non-Euclidean spherical geometries.

**Physical Model**: We build this framework upon the DD model from **"Diffusing diffusivity model of a polymer moving on a spherical surface"**, solving the coupled equations derived in this work.