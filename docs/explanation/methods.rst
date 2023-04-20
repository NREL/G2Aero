We focus on applications involving airfoil and blade design to emphasize the utility of the methods in a relatable environment where the separation of affine and undulation-type deformations is critical. 
However, we note obvious extensions to any tubular-like surface design, including but not limited to inlets, nozzles, ducts, passages, channels, etc., 
and encourage users to experiment with alternative types of swept surface designs.

Separable Shape Tensors (2D)
----------------------------
To define airfoil perturbations, we use Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis over matrix manifolds. 
We represent the 2D shape as an ordered sequence of :math:`n` landmarks :math:`(\mathbf{x}_i) \in \mathbb{R}^2, i=1, \dots, n` and :math:`n \geq 3`, 
which we can combine into a full-rank :math:`n`-by-:math:`2` matrix :math:`\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_n ]^\top` by concatenating 
the transposed landmarks down the rows of the matrix. We refer to these planar points as landmarks as they are often generated to 
define acceptable mesh resolution, quadrature nodes, or other desirable design refinements. The full rank restriction assumes that 
any generating 2D shape is not a point nor a line in the plane.

Through a separable tensor framework (:footcite:t:`Grey:2022`; :footcite:t:`Doronina:2022`), we decompose shapes into elements of the Grassmann manifold (Grassmannian) 
and relevant affine deformations (linear scaling plus translations). The deforming submanifolds of discrete 2D airfoil shapes as matrices 
:math:`\mathbf{X} \in  \mathbb{R}_*^{n \times  2}` (ignoring non-deformation translations) are subsequently separated into constituent 
parametrizations of undulation :math:`[\tilde{\mathbf{X}}] \in \mathcal{G}(n,2)` and linear :math:`\mathbf{P}` deformation types,

.. math:: \mathbf{X}(\mathbf{t},\mathbf{\ell}) = (\pi^{-1} \circ [\tilde{\mathbf{X}}])(\mathbf{t},\mathbf{\ell}) = \tilde{\mathbf{X}}(\mathbf{t})\mathbf{P}(\mathbf{\ell}).

The linear deformation portion is the :math:`2`-by-:math:`2` invertible matrix :math:`\mathbf{P} \in GL_2`, which stretches, rotates, and shears the shape 
into physically relevant scales/orientation over horizontal and vertical directions of the plane. Thus, in a complementary fashion, we utilize 
:math:`\mathcal{G}(n,2) \cong \mathbb{R}^{n\times 2}_*/GL_2` and :math:`\mathbf{\tilde{X}} \in \mathbb{R}^{n \times 2}_*` as the quotient topology of a 
full-rank representative element in an equivalence class :math:`[\mathbf{\tilde{X}}] \in \mathcal{G}(n,2)` of all matrices with equivalent span (:footcite:t:`Absil:2008`). 
As such, every element of the Grassmannian is a full-rank matrix modulo :math:`GL_2` deformations, and airfoil elements of the Grassmannian are decoupled 
from the aerodynamically constrained linear deformations :math:`\mathbf{P} \in GL_2` which predominantly control operational characteristics like 
angle-of-attack, thickness, and chordal scaling. 

Computations based on this formalism are predicated on a Landmark-Affine (LA) standardization which maps physical airfoil shapes :math:`\mathbf{X}` 
to (Stiefel representative) elements of the Grassmannian :math:`\mathbf{\tilde{X}}`. In particular, LA standardization normalizes the shape with 
zero mean and identity covariance. This standardization of the shape acts as a kind of scale normalization and is achieved by the singular 
value decomposition with computational complexity :math:`O(n)` or the related polar decomposition (:footcite:t:`Grey:2022`). 

To define airfoil perturbations, we use Principal Geodesic Analysis (PGA), a generalization of Principal Component Analysis over smooth 
manifolds (:footcite:t:`Fletcher:2003`). PGA is a data-driven approach which approximates a notion of the most amount of variability in a data 
set over projections onto a manifold. PGA determines principal components as elements in a central tangent space, 
:math:`T_{[\mathbf{\tilde{X}}_0]}\mathcal{G}(n,2)`, defined by input data where :math:`[\mathbf{\tilde{X}}_0]` is an intrinsic 
mean of the projected data over the manifold. PGA constitutes a manifold learning procedure for computing an important 
submanifold of :math:`\mathcal{G}(n,2)` representing a design space of relevant shapes inferred from provided data 
(:footcite:t:`Grey:2022`; :footcite:t:`Grey:2019`)---thus, data-driven manifold learning. Truncating the PGA basis expansions, we can study dramatic 
reductions in the total number of parameters :math:`\mathbf{t}` needed to describe any undulating perturbation to a representative 
2D shape as :math:`\tilde{\mathbf{X}}(\mathbf{t})`. This can then be balanced with given or inferred parametrizations of scale 
:math:`\mathbf{P}(\mathbf{\ell})` such that the dimension of :math:`\mathbf{\ell}` is no greater than four. The result is a framework 
for generating new samples from a data-driven submanifold of discrete 2D shapes. Example implementations are offered 
as iPython notebooks in the G2Aero repository.  

Riemannian Interpolation & Perturbation (3D)
--------------------------------------------

The mapping from airfoils to blades amounts to a smoothly varying set of affine deformations swept over discrete blade-span 
positions---a common convention in next-generation wind turbine blade design. The discrete blade can be represented by 
:math:`(\mathbf{X}_k)` as an ordered sequence of discrete airfoils with consistent :math:`n` planar landmarks such that :math:`k=1,\dots,N` 
are cross sections (airfoil landmarks)---thus constituting a sequence of sequences for structured mesh representations. 
However, the challenge is to interpolate these 2D shapes from potentially distinct airfoil classes to build a refined 
3D shape such that the interpolation preserves the desired affine deformations along the blade---e.g., chordal scaling 
composed with a twist over changing pitch axis.

The given sequence of discrete 2D shapes induces a sequence of equivalence classes :math:`([\tilde{\mathbf{X}}_k]) \in \mathcal{G}(n,2)` 
for :math:`k=1,...,N` at discrete blade-span positions :math:`\eta_k  \in  \mathcal{S} \subset  \mathbb{R}` 
to define a swept blade. The swept blade is defined as a piecewise geodesic path over the Grassmannian to interpolate 
discrete blade shapes independent of affine deformations. That is, we utilize a mapping 
:math:`\mathbf{\tilde{\gamma}}_{k,k+1}:[\tilde{\mathbf{X}}_k] \mapsto [\tilde{\mathbf{X}}_{k+1}]` as the geodesic 
interpolating from one representative standardized shape :math:`\tilde{\mathbf{X}}_k` to the next :math:`\tilde{\mathbf{X}}_{k+1}`. 
As a simple interpretation, a geodesic :math:`\mathbf{\tilde{\gamma}}_{k,k+1}` is the shortest path between two points of 
a manifold and represents a generalized notion of the "straight line" in this non-Euclidean topology of shapes. 
Thus, a full blade shape can be defined by interpolating standardized airfoil shapes using these 
piecewise geodesics over ordered blade-span positions :math:`\eta_k` along a non-Euclidean representative 
manifold of shapes. Finally, to get interpolated shapes back into physically relevant scales, 
we apply affine deformations, now including translations :math:`\mathbf{b}(\eta) \in \mathbb{R}^2`, 
based on previously constructed splines defining the carefully designed affine deformations,

.. math:: \mathbf{X}(\eta) = \mathbf{\tilde{X}}(\eta)\mathbf{P}(\eta)+\mathbf{1}\text{diag}(\mathbf{b}(\eta)).

An important caveat when inverting the shapes back to the physically relevant scales for subsequent twist and chordal deformations is Procrustes clustering. 
From the blade tip shape :math:`\tilde{\mathbf{X}}_{N}` to the blade hub shape :math:`\tilde{\mathbf{X}}_1`, we sequentially match the representative 
LA standardized shapes via Procrustes analysis. This offers rotations that can be applied to representative standardized airfoils for matching, 
which do not fundamentally modify the elements in the Grassmannian. Consequently, we cluster the sequence of representative shapes :math:`\tilde{\mathbf{X}}_k` 
by optimal rotations in each :math:`[\tilde{\mathbf{X}}_k]` to ensure they are best oriented from tip to hub to mitigate concerns about large variations 
in :math:`\mathbf{P}(\eta)`. This results in a natural framework for interpolating 2D shapes into swept definitions of 3D blades while simultaneously 
decoupling affine and higher-order undulation deformations. 

Lastly, blade perturbations are constructed from deformations to each of the given 2D cross sections over "consistent directions" randomly 
sampled at the central tangent space. Since a perturbation direction is defined by parameters :math:`\mathbf{t}` in the tangent space of the intrinsic 
(Karcher) mean, we utilize an isometry (preserving inner products) called parallel transport to smoothly "translate" the perturbing vector 
field along separate geodesics connecting the Karcher mean to each of the distinct ordered landmark airfoils along the swept surface. 
The result is a set of consistent directions---equal inner products and consequently equivalent "directions" :math:`\mathbf{t}` in the central 
tangent space---over ordered tangent spaces centered on each of the nominal :math:`([\tilde{\mathbf{X}}_k]) \in \mathcal{G}(n,2)` 
defining the blade. An example of a consistently perturbed sequence of airfoils to define a blade perturbation 
is offered as an iPython notebook example in the G2Aero repository. Finally, these consistently perturbed shapes are combined with 
three to six independently defined or inferred affine parameters to describe a rich feature space of 3D blade perturbations. 
Our impression is that this makes separable shape tensors a powerful tool enabling future aerodynamic design and swept 
tubular-like surface definitions. Evidence of these impressions is offered by successful applications of G2Aero to 
improve next-generation wind turbine blade design associated with the ARPA-E DIFFERENTIATE program.


.. bibliography::


.. footbibliography::