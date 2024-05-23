---
title: 'G2Aero: A Python package for separable shape tensors'
tags:
    - Python
    - airfoils
    - manifolds
    - generative model
    - shape representation
    - Grassmannian
    - data-driven
authors:
    - name: Olga A. Doronina
      orcid: 0000-0003-0872-7098
      corresponding: true
      affiliation: 1 # (Multiple affiliations must be quoted)
    - name: Zachary J. Grey
      affiliation: 2
    - name: Andrew Glaws
      affiliation: 1
affiliations:
    - name: National Renewable Energy Laboratory, USA
      index: 1

    - name: National Institute of Standards and Technology, USA
      index: 2

date: 29 November 2022
bibliography: paper.bib
---
# Summary

`G2Aero` is a Python package for the design and deformation of discrete planar curves and tubular surfaces using a geometric data-driven approach. `G2Aero` utilizes a topology of product manifolds: the Grassmannian, $\mathcal{G}(n,2)$---the set of $2$-dimensional subspaces in $\mathbb{R}^n$---and the symmetric positive-definite (SPD) manifold, $S^2_{+ +}$---the set of $2\times 2$ SPD matrices. The package provides a novel framework for representing separable deformations to shapes, which consist of stretching, scaling, rotating, and translating---also known as affine deformations---and a set of complementary deformations---which we refer to as undulation-type deformations.

We focus on airfoil and blade design applications to emphasize the utility of the methods in an environment where the separation of affine and undulation-type deformations is critical. Notable functionalities of the framework for blade design include:  

1) generating novel 2D (airfoil) shapes informed by a database of physically relevant airfoils,
2) building 3D blades by interpolating sequences of 2D airfoil cross-sections, and
3) generating blades with consistent perturbations along the blade span.

We discuss the framework and provide examples in the context of wind energy applications, specifically wind turbine blade design. \autoref{fig:wire_frame} shows the wire frame obtained by interpolating airfoils defining the IEA 15-MW wind turbine blade [@IEA15MW] and applying affine transformations corresponding to twist, chordal scaling, and bending. This, and all other figures in the paper, can be reproduced following examples and referencing supporting documentation provided in the `G2Aero` package.

# Statement of need

Aerodynamic shape design is a capstone problem in engineering and manufacturing that continues evolving with modern computational methods and resources. Many design and manufacturing algorithms rely on shape parameterizations to manipulate shapes to control and measure manufacturing and/or damage variations, develop surrogates, study sensitivities, approximate inverse problems, and inform optimizations.

![Blade wire-frame obtained by interpolating the colored cross-sections. Adapted from @Grey:2023.\label{fig:wire_frame}](wire_blade.png){ width=100% }

Two-dimensional cross-sections of aerodynamic structures such as intake passages, exhaust nozzles, bypass ducts, wings, gas turbine engine blades/vanes, and wind turbine blades are critical engineering shapes for aerospace and energy industries. Specifically, blade and wing design involves designing airfoils (2D cross-sections of a blade/wing) to improve aerodynamic, structural, and acoustic performance. Recent rapid development of artificial intelligence (AI) and machine learning (ML) algorithms made an airfoil design a growing area of research [@Li:2022] once again. Shape representations that better regularize deformations and reduce the dimension of the design space can have a significant impact in AI and ML applications [@glaws2022scitech; @grey2018active; @chen2020airfoil].

## Comparison with existing methods
There is a wide range of methods to represent airfoil geometry [@masters2017]. These methods vary from general geometry representation, such as B-splines [@Hosseini2016], to parametric methods specific to airfoil shapes, such as PARSEC [@sobieczky1999]. The current state-of-the-art method for airfoil shape parametrization is the class-shape transformation (CST) [@kulfan2008universal]. The parameters in this representation are coefficients of a truncated  Bernstein polynomial expansion and tuning them enables designers to define new airfoil shapes.

As an example, to generate new airfoils, researchers perturb CST coefficients of baseline airfoils (existing well-studied airfoils) and define design space as a hypercube with some empirical limits [@chen2020airfoil; @lim2019multi; @Glaws:2022]. But it can be challenging to define a broader, more general design space to generate new types of airfoils, e.g., using several baseline airfoils results in a very complicated and disjoint CST design space (see \autoref{fig:cst_vs_pga}a). The shape parametrization used in `G2Aero` is based on principal geodesic analysis (PGA) over the Grassmannian; it allows us to define an **improved low-dimensional parameter domain** for design and manufacturing algorithms. Details about PGA, as a generalization of Principal Component Analysis, are described in [@Fletcher:2003]. A random parameter sweep over the CST domain may produce non-physical airfoil shapes, while a random parameter sweep over the PGA domain results in "reasonable" (well-regularized) airfoils (\autoref{fig:cst_vs_pga}b).

![CST and PGA design spaces: a) 2D marginals of airfoil data, with colors indicating different classes of airfoils; b) airfoils obtained from random sweeps across CST and PGA domains. \label{fig:cst_vs_pga}](cst_vs_pga.png)

The CST method (and other explicit basis representations) often couples linear scaling of the shape (affine deformations) and undulating perturbations. Affine deformations---like changes in thickness, camber, and orientation---are often constrained in design problems (e.g., changes in thickness, Reynolds number, or angle-of-attack) and result in relatively well-understood physical impacts on aerodynamic performance; while undulating perturbations are of increasing interest to airfoil design [@glaws2022scitech; @grey2018active; @berguin2015method]. `G2Aero` **decouples linear scaling and undulations** by defining undulations as the set of all deformations modulo linear scaling of discrete curves. Because of this separability, we can independently study the effect of undulating perturbations---for example, manufacturing defects and damage.


Blade surfaces are generally defined by an ordered set of cross-sectional airfoils at different blade-span positions from hub to tip paired with profiles of twist, chordal scaling, and translation (bending). It can be challenging to interpolate an input sequence of 2D cross-sections along span-wise coordinates to define a 3D surface. Current approaches often require significant hand-tuning of airfoil shapes and interpolation methods to construct valid blade geometries without "dimples" or "kinks" in the blade's surface; e.g., families of airfoils defining modern wind turbine blades are carefully designed to ensure compatibility for blade definitions [@IEA15MW; @NREL5MW]. The chordal scaling profile represents linear deformations and is often tightly regulated by the operational conditions of the blade. Our separable treatment of linear scaling and undulations enables generalized surface interpolation, which is independent of these prescribed linear deformations and **generates more reliable/robust surface geometries** from diverse sets of airfoils potentially spanning distinct families.

Moreover, the number of parameters required to define an individual blade scales by the total number of designed cross-sections. For example, a wind turbine blade may be composed of eight to ten airfoil shapes along its span such that the total parameter count for the blade is an order of magnitude larger than the number of individual parameters defining an airfoil. Assuming each airfoil is defined independently, this could amount to hundreds of parameters required to represent a single blade shape. And the vast majority of these parameter combinations result in non-physical designs. `G2Aero` significantly **reduces the total number of parameters** by using parallel translation over the PGA domain to consistently perturb interpolated 2D shapes for 3D surface design (\autoref{fig:perturbed_blade}).

# Methods
`G2Aero` implements the novel separable tensor framework outlined in detail in @Grey:2023. It is designed to learn matrix-submanifolds, generate novel shapes localized to data, control shape undulations by truncating an ordered basis of deforming modes, and enable a novel notion of "consistent deformations" as an approach to regularized surface design.

Specific methods within `G2Aero` include:

- principal geodesic analysis (PGA) over normal coordinate neighborhoods of $\mathcal{G}(n,2)$ and $S^2_{+ +}$ matrix manifolds,
- Riemannian interpolation connecting a sequence of 2D cross-sections to build 3D swept surfaces from data,
- parallel translation over inferred PGA domain to perform consistent perturbations over the span of interpolated 2D shapes for 3D surface design.

![Consistent perturbations applied to all baseline airfoils of the IEA 15-MW blade.\label{fig:perturbed_blade}](perturbed_blade.png)

# Current capabilities and applications
### 1) Generating airfoil shapes

`G2Aero` defines an improved parameter domain inferred from a database of relevant shapes (discrete 2D curves). This domain independently treats affine and undulating type deformations, allowing for more targeted shape design and generation of a rich set of novel 2D deformations. Using PGA over the Grassmannian significantly reduces the dimensionality of the parameter domain. Using an extensive database of airfoils [@BigFoil:2022] and analyzing the shape reconstruction error, we found that we can use as few as four PGA parameters to represent a wide range of undulation-type deformations of the shape and two parameters to represent linear deformations.

In @Doronina:2022, we concisely summarized the framework and demonstrated the advantages of an improved parameter domain for ML/AI algorithms.
@zhang2022aerodynamic tested Grassmannian shape representation and demonstrated robustness for shape optimization applications. @jasa2022wind used airfoils generated by `G2Aero` coupled with NREL’s Wind-Plant Integrated System Design Engineering Model (WISDEM) [@wisdem] to design blade shapes with reduced costs of energy compared to traditional design methods.

### 2) Building 3D blades by interpolating 2D airfoil cross-sections

We use piecewise geodesic interpolation to connect a sequence of 2D cross-sections and build 3D swept surfaces from data. Our separable treatment of linear scaling and undulating deformations enables the generation of more reliable/robust swept surface geometries independent of often fixed operation conditions (twist, chord scaling, and bending of the blade). As part of `G2Aero` we provide an example script demonstrating how to generate a 3D computer-aided design (CAD) surface or surface mesh. We start from a YAML file in the format used to define a wind turbine blade. The blade shape definition contains ten airfoils at different blade-span positions and profiles of twist, chordal scaling, and bending. We generate 100 interpolated cross-sections defining a refined 3D surface. Then, using `Gmsh` [@geuzaine2009gmsh], we generate a 3D surface mesh shown on \autoref{fig:wind_blades}.

### 3) Generating perturbed blades

`G2Aero` achieves flexibility to generate designs in a customizable way over any portion of the blade such that deformations are independent of the deformations governing the operational conditions (twist, chordal scaling, and bending of the blade). The blade perturbation capability of `G2Aero` has been successfully used in @glaws_invertible_2022.

![Structured surface mesh of the IEA 15-MW blade. Adapted from from @Grey:2023\label{fig:wind_blades}](wind_blades.png)

# Outlook and Future Work

Despite our focus on airfoil and blade design, we note obvious extensions to a variety of 2D shape applications as well as 3D shapes that are well-defined by a sequence of cross-sections, such as inlets, nozzles, ducts, passages, channels, etc. We encourage users to experiment with `G2Aero` and apply methods to alternative types of shapes and surfaces.  Our computationally efficient approach to matrix-manifold learning and generative modeling of discrete planar curves may offer advantages in applications beyond aerodynamics, where learning a non-linear topology of shapes from data may be impactful.

# Acknowledgments

Certain software are identified in this paper in order to specify the experimental procedure adequately.  Such identification is not intended to imply recommendation or endorsement of any product or service by NIST, nor is it intended to imply that the software identified are necessarily the best available for the purpose.

This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding is partially provided by the Advanced Research Projects Agency-Energy (ARPA-E) Design Intelligence Fostering Formidable Energy Reduction and Enabling Novel Totally Impactful Advanced Technology Enhancements (DIFFERENTIATE) program. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. This work is U.S. Government work and not protected by U.S. copyright. A portion of this research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory.

# References  
