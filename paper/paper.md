
  

---

title: 'G2Aero: A Python package for separable shape tensors'

tags:

- Python

  

authors:

- name: Olga A. Doronina^[corresponding author]

orcid: 0000-0003-0872-7098

affiliation: 1 # (Multiple affiliations must be quoted)

- name: Zachary Grey

affiliation: 2

- name: Andrew Glaws

affiliation: 1

affiliations:

- name: National Renewable Energy Laboratory

index: 1

- name: National Institute of Standards and Technology

index: 2

  

date: 22 November 2022

bibliography: paper.bib

---

# Summary

Aerodynamic shape design is a capstone problem in engineering and manufacturing which continues to evolve with modern computational resources. Many design and manufacturing algorithms rely on shape parametrizations as representations to manipulate shapes in order to control and measure manufacturing and/or damage variations, develop surrogates, study sensitivities, approximate inverse problems, and inform optimizations. Airfoil (2D shape) design is often the first step in designing an entire blade/wing as a 3D surface or volume. Our presentation is primarily concerned with the aerodynamic implications of wind turbine blade design but extensions to alternative applications are conceivable. 

As an example, the blade design process consists of systematically designing distinct airfoils as cross sections of the 3D surface and selecting operational conditions---prescribed by individual scales and orientations of airfoils in a global reference frame---which must be tightly controlled down the span of the blade or wing to satisfy structural and/or legal requirements. The fundamental mathematical challenge is how to best interpolate these often distinct 2D cross section designs along the span of the 3D surface such that the governing operational conditions are retained or independently parametrized. Beyond this specific application, there are a number of extensions to tubular-like shapes and swept surface definitions which may also benefit from the sought separable forms.

In keeping with the candidate airfoil and blade design challenges, we describe a computational framework for data-driven discrete shape representations to connect workflows for 2D airfoil designs with 3D blade designs. This perspective facilitates a novel approach to regularize airfoil and blade perturbations which are---by definition---independent of deformations governing operational characteristics for each cross section. Moreover, this novel interpretation offers a set of reduced blade parameters which are independent of the total number of designed cross sections. The framework, `G2Aero`, is a Python package for the design and deformation of 2D shapes and 3D surfaces using a geometric data-driven approach. `G2Aero` utilizes a topology of matrix manifolds---principally the Grassmannian, $\mathcal{G}(n,2)$---to build up a novel framework for representing separable deformations to shapes. `Consequently, G2Aero` achieves this flexibility to generate designs in a customizable way over any portion of the blade such that deformations are independent of the deformations governing the operational conditions. 

Our example data-driven methods are based on a curated database of diverse geometrically relevant airfoils. Specific methods within `G2Aero` include: (i) principal geodesic analysis over normal coordinate neighborhoods of matrix manifolds, (ii) a variety of data-regularized deformations to nominal 2D shapes, (iii) Riemannian interpolation connecting a sequence of 2D cross sections to build 3D swept surfaces from data, and (iv) consistent perturbations over the span of interpolated 2D shapes for 3D surface design. Given the aforementioned methods, notable functionalities of this framework for blade design include: (i) generating novel separable airfoil shapes informed by a database of physically relevant airfoils, (ii) building 3D blades by interpolation sequences of 2D airfoil cross sections, (iii) generating perturbed blades with "consistent" airfoil perturbations for dramatic dimension reduction in blade design, and (iv) extensibility beyond blade and wing design.

# Statement of need

Two-dimensional cross sections of aerodynamic structures such as intake passages, exhaust nozzles, bypass ducts, aircraft wings, gas turbine engine blades/vanes, and wind turbine blades are critical engineering tubular-like shapes whose design and manufacturing can have significant impacts on the aerospace and energy industries. Specifically, for blade and wing design as a relatable example, these cross sections become airfoils. Research into AI and ML algorithms involving airfoil design for improved aerodynamic, structural, and acoustic performance is a rapidly growing area of work (Zhang, Sung, and Mavris 2018; Li, Bouhlel, and Martins 2019; Chen, Chiu,and Fuge 2019; Glaws et al. 2021; Jing et al. 2021; Yonekuraand Suzuki 2021; Yang, Lee, and Yee 2021).

Resonating with applications involving airfoil design, the current state-of-the-art for airfoil shape parametrization is the class-shape transformation (CST) method. The parameters in this representation are coefficients of a polynomial expansion and tuning them enables designers to define new airfoil shapes. However, defining a meaningful design space of CST parameters across a collection of diverse airfoil types is difficult. That is, it is challenging to interpret how modified CST parameters will perturb the shape and thus difficult to contain or bound CST parameters to produce "reasonable" (well-regularized) aerodynamic shapes. Furthermore, CST representations couple affine-type deformations---critical to controlling operational conditions---with a rich set of undulating-type perturbations. Affine deformations result in significant and relatively well-understood impacts to aerodynamic performance which are used to establish the operational conditions of blade cross sections. In a complementary fashion, undulating perturbations are characteristic of manufacturing variations and blade damage/soiling that are of increasing interest to airfoil designers across industries---particularly in the design of off-shore wind turbines. Separability between physically meaningful affine deformations and higher-order undulations in shapes is a desirable feature of the shape design process. Moreover, any specific airfoil parametrization may not be extensible to alternative applications involving more general tubular-like surface design.

In 3D extensions, individual airfoil parametrizations representing 2D cross sections in blade and wing designs scale the individual airfoil parameter count by a multiple of the total number of designed cross sections---e.g., often eight to ten airfoils may be designed to represent a single wind turbine blade thus, at most, increasing the total parameter count for the blade by an order of magnitude over the number of individual airfoil parameters. (Assuming each airfoil is distinct, for example, this could amount to hundreds of parameters to represent a single blade.) Additionally, it is not immediately clear how to best interpolate parameters defining a relatively sparse total number of 2D cross sections for refinement of a 3D surface definition along the span axis. Moreover, alternative approaches to surface representation often lack the intuitive extension of 2D airfoil design for 3D surface construction such that each 2D design is intuitively "swept" along a span axis of the 3D surface. These surface representations, such as free-form deformation, are often flexible but complicate a workflow predicated on extending 2D designs for 3D definitions. The nature of the "free-form" couples deformation types from one cross section to the next, often inadvertently perturbing neighboring airfoil operational characteristics spanned by a common volumetric cube or hull.

`G2Aero` defines an improved coordinate framework for arbitrary tubular cross sectional shape design given a database of relevant shapes which separates affine and undulating type deformations. This new separable representation offers (i) a rich set of novel 2D deformations separated into linear and high-order undulation-type deformations, and (ii) an improved low-dimensional parameter domain for inferential statistics informing design and manufacturing. 

In summary, specific to wind energy, 3D blade designs characterized by an ordered set of planar airfoils at different blade-span positions from hub to tip along the span are paired with profiles of twist, chord scaling, and translation (bend) defining the operational conditions. Current approaches to blade design require significant hand-tuning of airfoil parameters to ensure the construction of valid blade geometries without "dimples" or "kinks" in the surface of the blade. Our separable treatment of these deformations types in `G2Aero` enables the automatic generation of more reliable/robust swept surface geometries by separating parametrizations of either type of deformation. As an added benefit, `G2Aero` is extensible to the flexible design of general tubular cross sections by applying consistent universal parameter deformations to all 2D shapes and improved interpolation of these 2D shapes for more robust and tractable 3D surface design.

# Methods

We focus on applications involving airfoil and blade design to emphasize the utility of the methods in a relatable environment where separation of affine and undulation type deformations is critical. However, we note obvious extensions to any tubular-like surface design including but not limited to inlets, nozzles, ducts, passages, channels, etc. and encourage users to experiment with alternative types of swept surface designs.

## Separable Shape Tensors (2D)
To define airfoil perturbations we use Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis over matrix manifolds. We represent the 2D shape as an ordered sequence of $n$ landmarks $(\mathbf{x}_i) \in \mathbb{R}^2, i=1, \dots, n$ and $n \geq 3$, which we can combine into a full-rank $n$-by-$2$ matrix $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n ]^\top$ by concatenating the transposed landmarks down the rows of the matrix. We refer to these planar points as landmarks as they are often generated with the intent of defining acceptable mesh resolution, quadrature nodes, or other desirable design refinements. The full rank restriction assumes that any generating 2D shape is not a point nor a line in the plane. 

Through a separable tensor framework, (Grey 2022; Doronina 2022) we decompose shapes into elements of the Grassmann manifold (Grassmannian) and relevant affine deformations (linear scaling plus translations).  The deforming submanifolds of discrete 2D airfoil shapes as matrices $\mathbf{X} \in  \mathbb{R}_*^{n \times  2}$ (ignoring non-deformation translations) are subsequently separated into constituent parametrizations of undulation $[\tilde{\mathbf{X}}] \in \mathcal{G}(n,2)$ and linear $\mathbf{P}$ deformation types,
$$
\mathbf{X}(\mathbf{t},\mathbf{\ell}) = (\pi^{-1} \circ [\tilde{\mathbf{X}}])(\mathbf{t},\mathbf{\ell}) = \tilde{\mathbf{X}}(\mathbf{t})\mathbf{P}(\mathbf{\ell}).
$$
The linear deformation portion is the $2$-by-$2$ invertible matrix $\mathbf{P} \in GL_2$ which stretches, rotates, and shears the shape into a physically relevant scales/orientation over horizontal and vertical directions of the plane. Thus, in a complementary fashion, we utilize $\mathcal{G}(n,2) \cong \mathbb{R}^{n\times 2}_*/GL_2$ and $\mathbf{\tilde{X}} \in \mathbb{R}^{n \times 2}_*$ as the quotient topology of a full-rank representative elements in an equivalence class $[\mathbf{\tilde{X}}] \in \mathcal{G}(n,2)$ of all matrices with equivalent span (Absil 2008). As such, every element of the Grassmannian is a full-rank matrix modulo $GL_2$ deformations, and airfoil elements of the Grassmannian are decoupled from the aerodynamically constrained linear deformations $\mathbf{P} \in GL_2$ which predominantly control operational characteristics like angle-of-attack, thickness, and chordal scaling. 

Computations based on this formalism are predicated on a Landmark-Affine (LA) standardization which maps physical airfoil shapes $\mathbf{X}$ to (Stiefel representative) elements of the Grassmannian $\mathbf{\tilde{X}}$. In particular, LA standardization normalizes the shape such that it has zero mean and identity covariance. This standardization of the shape acts as a kind of scale-normalization and is achieved by the singular value decomposition with computational complexity $O(n)$ or the related polar decomposition (Grey 2022). 

To define airfoil perturbations, we use Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis over smooth manifolds (Fletcher et al. 2003). PGA is a data-driven approach which approximates a notion of the most amount of variability in a data set over projections onto a manifold. PGA determines principal components as elements in a central tangent space, $T_{[\mathbf{\tilde{X}}_0]}\mathcal{G}(n,2)$, defined by input data where $[\mathbf{\tilde{X}}_0]$ is an intrinsic mean of the projected data over the manifold. PGA constitutes a manifold learning procedure for computing an important submanifold of $\mathcal{G}(n,2)$ representing a design space of relevant shapes inferred from provided data (Grey 2022; Grey 2019)---thus, data-driven manifold learning. Truncating the PGA basis expansions, we can study dramatic reductions in the total number of parameters $\mathbf{t}$ needed to describe any undulating perturbation to a representative 2D shape as $\tilde{\mathbf{X}}(\mathbf{t})$. This can then be balanced with given or inferred parametrizations of scale $\mathbf{P}(\mathbf{\ell})$ such that the dimension of $\mathbf{\ell}$ is no greater than four. The result is a framework for generating new samples from a data-driven submanifold of discrete 2D shapes. Example implementations are offered as iPython notebooks in the `G2Aero` repository.  

## Riemannian Interpolation & Perturbation (3D)

The mapping from airfoils to blades amounts to a smoothly varying set of affine deformations swept over discrete blade-span positions---a common convention in next-generation wind turbine blade design. The discrete blade can be represented by $(\mathbf{X}_k)$ as an ordered sequence of discrete airfoils with consistent $n$ planar landmarks such that $k=1,\dots,N$ are cross sections (airfoil landmarks)---thus constituting a sequence of sequences for structured mesh representations. However, the challenge is to interpolate these 2D shapes from potentially distinct airfoil classes to build a refined 3D shape such that the interpolation preserves the desired affine deformations along the blade---e.g., chordal scaling composed with twist over changing pitch axis.

The given sequence of discrete 2D shapes induces a sequence of equivalence classes $([\tilde{\mathbf{X}}_k]) \in  \mathcal{G}(n,2)$ for $k=1,...,N$ at discrete blade-span positions $\eta_k  \in  \mathcal{S} \subset  \mathbb{R}$ to define a swept blade. The swept blade is defined as a piecewise geodesic path over the Grassmannian to interpolate discrete blade shapes independent of affine deformations. That is, we utilize a mapping $\mathbf{\tilde{\gamma}}_{k,k+1}:[\tilde{\mathbf{X}}_k] \mapsto [\tilde{\mathbf{X}}_{k+1}]$ as the geodesic interpolating from one representative standardized shape $\tilde{\mathbf{X}}_k$ to the next $\tilde{\mathbf{X}}_{k+1}$. As a simple interpretation, a geodesic $\mathbf{\tilde{\gamma}}_{k,k+1}$ is the shortest path between two points of a manifold and represents a generalized notion of the "straight line" in this non-Euclidean topology of shapes. Thus, a full blade shape can be defined by interpolating standardized airfoil shapes using these piecewise-geodesics over ordered blade-span positions $\eta_k$ along a non-Euclidean representative manifold of shapes. Finally, to get interpolated shapes back into physically relevant scales, we apply affine deformations, now including translations $\mathbf{b}(\eta) \in \mathbb{R}^2$, based on previously constructed splines defining the carefully designed affine deformations,
$$
\mathbf{X}(\eta) = \mathbf{\tilde{X}}(\eta)\mathbf{P}(\eta)+\mathbf{1}\text{diag}(\mathbf{b}(\eta)).
$$

An important caveat when inverting the shapes back to the physically relevant scales for subsequent twist and chordal deformations is a Procrustes clustering. From the blade tip shape $\tilde{\mathbf{X}}_{N}$ to the blade hub shape $\tilde{\mathbf{X}}_1$, we sequentially match the representative LA standardized shapes via Procrustes analysis. This offers rotations that can be applied to representative standardized airfoils for matching which do not fundamentally modify the elements in the Grassmannian. Consequently, we cluster the sequence of representative shapes $\tilde{\mathbf{X}}_k$ by optimal rotations in each $[\tilde{\mathbf{X}}_k]$ to ensure they are best oriented from tip to hub to mitigate concerns about large variations in $\mathbf{P}(\eta)$. This results in a natural framework for interpolating 2D shapes into swept definitions of 3D blades while simultaneously decoupling affine and higher-order undulation deformations. 

Lastly, blade perturbations are constructed from deformations to each of the given 2D cross sections over "consistent directions" randomly sampled at the central tangent space. Since a perturbation direction is defined by parameters $\mathbf{t}$ in the tangent space of the intrinsic (Karcher) mean, we utilize an isometry (preserving inner products) called parallel transport to smoothly “translate” the perturbing vector field along separate geodesics connecting the Karcher mean to each of the distinct ordered landmark airfoils along the swept surface. The result is a set of consistent directions---equal inner products and consequently equivalent "directions" $\mathbf{t}$ in the central tangent space---over ordered tangent spaces centered on each of the nominal $([\tilde{\mathbf{X}}_k]) \in  \mathcal{G}(n,2)$ defining the blade. An example of a consistently perturbed sequence of airfoils to define a blade perturbation is offered as an iPython notebook example in the `G2Aero` repository. Finally, these consistently perturbed shapes are combined with three to six independently defined or inferred affine parameters to describe a rich feature space of 3D blade perturbations. Our impression is that this makes separable shape tensors a powerful tool enabling future aerodynamic design and swept tubular-like surface definitions. Evidence of these impressions is offered by successful applications of `G2Aero` to improve next generation wind turbine blade design associated with the ARPA-E DIFFERENTIATE program.

# Acknowledgments

This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding partially provided by the Advanced Research Projects Agency-Energy (ARPA-E) Design Intelligence Fostering Formidable Energy Reduction and Enabling Novel Totally Impactful Advanced Technology Enhancements (DIFFERENTIATE) program. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. This work is U.S. Government work and not protected by U.S. copyright. A portion of this research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory.
