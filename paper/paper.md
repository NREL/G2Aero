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

date: 20 December 2021
bibliography: paper.bib
---
  
# Summary  
Aerodynamic shape design is a classical problem in engineering and manufacturing. In particular, many AI-aided design and manufacturing algorithms rely on shape parametrization methods to manipulate shapes in order to study sensitivities, approximate inverse problems, and inform optimizations. Airfoil (2D shape) design is usually the first step in designing an entire blade/wing as 3D surface or volume. We describe a systematic framework using a novel representation of blades to connect workflows for 2D airfoil design to 3D blades/wings.
  
`G2Aero` is a Python package for design and deformation of 2D airfoils and 3D blades using a data-driven approach. We utilize the geometry of matrix manifolds---specifically the Grassmannian, $\mathcal{G}(n,2)$---to build a novel framework for representing physics-based separable deformations of shapes. `G2Aero` offers the flexibility to generate perturbations in a customizable way over any portion of the blade. The `G2Aero` framework utilizes data-driven methods based on a curated database of physically relevant airfoils. Specific methods include: (i) principal geodesic analysis over normal coordinate neighborhoods of matrix manifolds, (ii) a variety of data-regularized deformations to nominal 2D airfoil shapes, (iii) Riemannian interpolation connecting a sequence of airfoil cross-sections to build 3D blades from 2D data, and (iv) consistent perturbations over the span of interpolated 3D blades based on dominant modes from the data-driven analysis. Given the above methods, notable functionalities of this framework include: (i) generating novel separable airfoil shapes informed by a database of physically relevant airfoils, (ii) building 3D blades by interpolation sequences of 2D airfoil cross sections, and (iii) generating perturbed blades based with "consistent" airfoil perturbations for dramatic dimension reduction in blade design.

# Statement of need
Two-dimensional cross sections of aerodynamic structures such as aircraft wings or wind turbine blades, also known as airfoils, are critical engineering shapes whose design and manufacturing can have significant impacts on the aerospace and energy industries. Research into AI and ML algorithms involving airfoil design for improved aerodynamic, structural, and acoustic performance is a rapidly growing area of work (Zhang, Sung, andMavris 2018; Li, Bouhlel, and Martins 2019; Chen, Chiu,and Fuge 2019; Glaws et al. 2021; Jing et al. 2021; Yonekuraand Suzuki 2021; Yang, Lee, and Yee 2021).

The current state-of-the-art for airfoil shape parametrization is the class-shape transformation (CST) method [@kulfan2008universal]. The parameters in this representation are coefficients of the polynomial expansion and tuning them allow designers to define new airfoil shapes. However, defining a meaningful design space of CST parameters across a collection of diverse airfoil types is difficult. That is, it is challenging to interpret how modified CST parameters will perturb the shape and thus difficult to contain or bound CST parameters to produce "reasonable" (well-regularized) aerodynamic shapes. Furthermore, CST representations couple affine-type deformations with undulating-type perturbations. Affine deformations result in significant and relatively well-understood impacts to aerodynamic performance. In the complement, undulating perturbations are characteristic of manufacturing variations and blade damage/soiling that are of increasing interest to airfoil designers across industries---particularly in the design of off-shore wind turbines. This coupling between physically meaningful affine deformations---critical for aerodynamic operational characteristics of blade definitions---and higher-order undulations in shapes complicates the design process.

`G2Aero` defines an improved coordinate framework for airfoil and blade design given a database of relevant airfoils which decouples the two aforementioned deformation types. This new separable representation offers (i) a rich set of novel 2D airfoil deformations separated into linear and high-order undulation-type deformations, and (ii) an improved low-dimensional parameter domain for inferential statistics informing design and manufacturing. In the context of wind energy, 3D blade designs are often characterized by an ordered set of planar airfoils at different blade-span positions from hub to tip of the blade as well as profiles of twist, chord scaling, and translation. Current approaches to blade design require significant hand-tuning of airfoil parameters to ensure the construction of valid blade geometries without "dimples" or "kinks" in the surface of the blade. Our new separable treatment of these deformations types enables the automatic generation of more reliable geometries by separating parametrizations of either type of deformation.
  
`G2Aero` enables the flexible design of novel airfoils and blades by applying consistent universal parameter deformations to all airfoils and smooth interpolation of shapes between landmarks.

# Methods  

## 2D Airfoil perturbations  
To define airfoil perturbations we use Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis over matrix manifolds. In our case, we define submanifolds of discrete 2D airfoil shapes as matrices $\mathbf{X}$ separated into constituent undulation and linear deformation type pieces, 

\begin{equation} \label{eq:separable_shape}
  \mathbf{X}(\mathbf{t},\mathbf{\ell}) = (\pi^{-1} \circ [\tilde{\mathbf{X}}])(\mathbf{t},\mathbf{\ell}) = \tilde{\mathbf{X}}(\mathbf{t})\mathbf{P}(\,mathbf{\ell})$.
\end{equation}

PGA is a data-driven approach which approximates a notion of the most amount of variability in a given data set over projections onto a manifold. Truncating the PGA basis expansions, we can reduce the number of parameters needed to describe any perturbation to a shape. Examples are offered as iPython notebooks in the `G2Aero` repository.

## 3D blade interpolation
The mapping from airfoils to blades amounts to a smoothly varying set of affine deformations over discrete blade-span positions---a common convention in next-generation wind turbine blade design. The discrete blade can be represented as a sequence of matrices $(\mathbf{X}_k) \in \mathbb{R}_*^{n\times2}$ for $k=1,\dots,N$. However, the challenge is to interpolate these shapes from potentially distinct airfoil classes to build a refined 3D shape such that the interpolation preserves the desired affine deformations along the blade (chordal scaling composed with twist over changing pitch axis).

Given an induced sequence of equivalence classes $([\mathbf{\tilde{X}}_k]) \in \mathcal{G}(n,2)$ for $k=1,...,N$ at discrete blade-span positions $\eta_k \in \mathcal{S} \subset \mathbb{R}$ from a given blade definition, we can construct a piecewise geodesic path over the Grassmannian to interpolate discrete blade shapes independent of affine deformations. That is, we utilize a mapping $\mathbf{\tilde{\gamma}}_{k,k+1}:[\mathbf{\tilde{X}}_k] \mapsto [\mathbf{\tilde{X}}_{k+1}]$ as the geodesic interpolating from one representative standardized shape $\tilde{\mathbf{X}$ to the next~\cite{edelman1998geometry}. As a simple interpretation, a geodesic $\mathbf{\tilde{\gamma}}_{k,k+1}$ is the shortest path between two points of a manifold and represents a generalized notion of the "straight line" in this non-linear topology of shapes. Thus, a full blade shape can be defined by interpolating standardized airfoil shapes using these piecewise-geodesics over ordered blade-span positions $\eta_k$ along a non-linear representative manifold of shapes. Finally, to get interpolated shapes back into physically relevant scales, we apply inverse affine transformation based on previously constructed splines defining the carefully designed affine deformations,

\begin{equation} \label{eq:blade}
	\mathbf{X}(\eta) = \mathbf{\tilde{X}}(\eta)\mathbf{M}(\eta)+\mathbf{1}\text{diag}(\mathbf{b}(\eta)).
\end{equation}

An important caveat when inverting the shapes in~\eqref{eq:blade} back to the physically relevant scales for subsequent twist and chordal deformations is a \emph{Procrustes clustering}. From the blade tip shape $\mathbf{\tilde{X}}_{N}$ to the blade hub shape $\mathbf{\tilde{X}}_1$, we sequentially match the representative LA standardized shapes via Procrustes analysis~\cite{gower1975generalized}. This offers rotations that can be applied to representative standardized airfoils for matching---which do not fundamentally modify the elements in the Grassmannian. Consequently, we cluster the sequence of representative shapes $\mathbf{\tilde{X}}_k$ by optimal rotations in each $[\mathbf{\tilde{X}}_k]$ to ensure they are best oriented from tip to hub to mitigate concerns about large variations in $\mathbf{M}(\eta)$.

## 3D blade perturbations
Blade perturbations are constructed from deformations to each of the given airfoil cross sections over consistent directions randomly sampled at the central tangent space. Since a perturbation direction is defined in the tangent space of Karcher mean, we utilize an isometry (preserving inner products) called parallel transport to smoothly “translate” the perturbing vector field along separate geodesics connecting the Karcher mean to each of the distinct individual ordered airfoils. The result is a set of consistent directions (equal inner products and consequently equivalent normal coordinates in the central tangent space) over ordered tangent spaces centered on each of the nominal $([\mathbf{\tilde{X}}_k]) \in \mathcal{G}(n,2)$ defining the blade. An example of a consistently perturbed sequence of airfoils to define a blade perturbation is offered as an iPython notebook example in the `G2Aero` repository. Finally, these consistently perturbed shapes are combined with three to six independent affine parameters to describe a rich feature space of 3D blade perturbations.
  
# Acknowledgments  
This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding partially provided by the Advanced Research Projects Agency-Energy (ARPA-E) Design Intelligence Fostering Formidable Energy Reduction and Enabling Novel Totally Impactful Advanced Technology Enhancements (DIFFERENTIATE) program. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. This work is U.S. Government work and not protected by U.S. copyright. A portion of this research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory.
  
# References  




