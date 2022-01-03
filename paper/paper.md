---
title: 'G2Aero: A Python package for Grassmannian shape representation in aerodynamic applications'
tags:
  - Python

authors:
  - name: Olga A. Doronina^[corresponding author]
    orcid: 0000-0003-0872-7098
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Zachary Grey
    affiliation: 2
affiliations:
 - name: National Renewable Energy Laboratory
   index: 1
 - name: National Institute of Standards and Technology
   index: 2

date: 20 Decenber 2021
bibliography: paper.bib
---
  
# Summary  
Aerodynamic shape design is a classical problem in engineering and manufacturing. In particular, many AI-aided design and  manufacturing algorithms rely on shape parameterization methods to manipulate shapes in order to study sensitivities,   
approximate inverse problems, and inform optimizations. Airfoil (2D shape) design is usually used as a step to full blade/wing (3D shape) design. 
  
`G2Aero` is a Python package for Grassmannian shape representation. It's functionalities include: 1) generating perturbed airfoils informed by a database of physically relevant airfoils; 2) interpolating 3d blade from 2D cross-sections; 3) generating perturbed blades based on 'consistent' airfoil perturbation along the blade.
# Statement of need  (clearly illustrates the research purpose of the software)
Two-dimensional cross-sections of aerodynamic structures such as aircraft wings or wind turbine blades, also known as airfoils, are critical engineering shapes whose design and manufacturing can have significant impacts on the aerospace and energy industries. Research into AI and ML algorithms involving airfoil design for improved aerodynamic, structural, and acoustic performance is a rapidly growing area of work (Zhang, Sung, andMavris 2018; Li, Bouhlel, and Martins 2019; Chen, Chiu,and Fuge 2019; Glaws et al. 2021; Jing et al. 2021; Yonekuraand Suzuki 2021; Yang, Lee, and Yee 2021).

The current state-of-the-art for airfoil shape parametrization is the class-shape transformation (CST) method [@kulfan2008universal]. The parameters in this representation are coefficients of the polynomial expansion and tuning them allow one to define new  airfoil shapes. However, defining a meaningful design space of CST parameters across a collection of airfoil types is difficult. That is, it is challenging to interpret how modified CST parameters will perturb the shape and thus difficult to contain or bound CST parameters to produce "reasonable" aerodynamic shapes. Furthermore, CST representations couple large-scale affine-type deformations---deformations resulting in significant and relatively well-understood impacts to aerodynamic  performance---with undulating perturbations that are of increasing interest to airfoil designers across industries. This coupling between physically meaningful affine deformations and undulations in shapes resulting from higher-order polynomial perturbations complicates the design process.

`G2Aero` define airfoil parametrization given a data-set of relevant airfoils. This new parametrization offers (i) a rich set of novel 2D airfoil deformations not previously captured in the data, (ii) improved low-dimensional parameter domain for inferential statistics informing design/manufacturing

The Grassmannian framework for airfoil representation has the additional benefit of enabling the design of three-dimensional wings and blades. In the context of wind energy, full blade designs are often characterized by an ordered set of planar airfoils at different blade-span positions from hub to tip of the blade as well as  profiles of twist, chord scaling, and translation. Current approaches to blade design require significant hand-tuning of airfoils to ensure the construction of valid blade geometries without dimples or kinks. 

`G2Aero` 
  
`G2Aero` enables the flexible design of new blades by applying consistent deformations to all airfoils and smooth interpolation of shapes between landmarks.
# Methods  

## 2D Airfoil perturbations  
To define airfoil perturbations we use Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis over smooth manifolds (in our case, a space of 2D airfoil shapes). PGA is a data-driven approach, which attempts to describe the most amount of variability of a given data-set over a manifold. 

Truncating the principal vector basis, we can reduce the number of parameters needed to describe a perturbation to the shape. 
## 3D blade interpolation
The mapping from airfoils to blades amounts to a smoothly varying set of affine deformations over discrete blade-span positions---a common convention in next-generation wind turbine blade design. The discrete blade can be represented as a sequence of matrices $(\bm{X}_k) \in \mathbb{R}_*^{n\times2}$ for $k=1,\dots,N$. However, the challenge is to interpolate these shapes from potentially distinct airfoil classes to build a refined 3D shape such that the interpolation preserves the desired affine deformations along the blade (chordal scaling composed with twist over changing pitch axis).

Given an induced sequence of equivalence classes $([\bm{\tilde{X}}_k]) \in \mathcal{G}(n,2)$ for $k=1,...,N$ at discrete blade-span positions $\eta_k \in \mathcal{S} \subset \mathbb{R}$ from a given blade definition (see the colored curves in Figure~\ref{fig:interp_blade}), we can construct a piecewise geodesic path over the Grassmannian to interpolate discrete blade shapes independent of affine deformations. That is, we utilize a mapping $\bm{\tilde{\gamma}}_{k,k+1}:[\bm{\tilde{X}}_k] \mapsto [\bm{\tilde{X}}_{k+1}]$ as the geodesic interpolating from one representative LA-standardized shape to the next~\cite{edelman1998geometry}.\footnote{A geodesic $\bm{\tilde{\gamma}}_{k,k+1}$ is the shortest path between two points of a manifold and represents a generalized notion of the "straight line" in this non-linear topology.} Thus, a full blade shape can be defined by interpolating LA-standardized airfoil shapes using these piecewise-geodesics over ordered blade-span positions $\eta_k$ along a non-linear representative manifold of shapes. Finally, to get interpolated shapes back into physically relevant scales, we apply inverse affine transformation based on previously constructed splines defining the carefully designed affine deformations,
\begin{equation} \label{eq:blade}
	\bm{X}(\eta) = \bm{\tilde{X}}(\eta)\bm{M}(\eta)+\bm{1}\text{diag}(\bm{b}(\eta)).
\end{equation}

An important caveat when inverting the shapes in~\eqref{eq:blade} back to the physically relevant scales for subsequent twist and chordal deformations is a \emph{Procrustes clustering}. From the blade tip shape $\bm{\tilde{X}}_{N}$ to the blade hub shape $\bm{\tilde{X}}_1$, we sequentially match the representative LA-standardized shapes via Procrustes analysis~\cite{gower1975generalized}. This offers rotations that can be applied to representative LA-standardized airfoils for matching---which do not fundamentally modify the elements in the Grassmannian. Consequently, we cluster the sequence of representative shapes $\bm{\tilde{X}}_k$ by optimal rotations in each $[\bm{\tilde{X}}_k]$ to ensure they are best oriented from tip to hub to mitigate concerns about large variations in $\bm{M}(\eta)$.
## 3D blade perturbations
Blade perturbations are constructed from deformations to each of the given cross-sectional airfoils inconsistent directions over t∈T0A4. Since a perturbation direction is defined in the tangent space of Karcher mean, we utilize an isometry (preserving inner products) called parallel transport to smoothly “translate” the perturbing vector field along separate geodesics connecting the Karcher mean to each of the individual ordered airfoils. The result is a set of consistent directions (equal inner products and consequently equivalent normal coordinates in the central tangent space) over ordered tangent spaces T[ ̃Xk]G(n,2)centered on each of the nominal[ ̃Xk] defining the blade. An example of consistently perturbed sequence of cross-sectional airfoils is shown in Figure 2. Finally, these four principal components are combined with three to six independent affine parameters constituting a full set of 7-10 parameters describing a rich feature space of 3D blade perturbations.
  
# Acknowledgments  
This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding partially provided by the Advanced Research Projects Agency-Energy (ARPA-E) Design Intelligence Fostering Formidable Energy Reduction and Enabling Novel Totally Impactful Advanced Technology Enhancements (DIFFERENTIATE) program. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. This work is U.S. Government work and not protected by U.S. copyright. A portion of this research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory.
  
# References  




