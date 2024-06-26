Welcome to G2Aero's documentation!
==================================

G2Aero is a flexible and practical tool for design and deformation of 2D airfoils and 3D blades using data-driven approaches. 
G2Aero utilizes the geometry of matrix manifolds -- specifically the Grassmannian -- to build a novel framework for representing physics-based separable deformations of shapes. 
G2Aero offers the flexibility to generate perturbations in a customizable way over any portion of the blade. 
The G2Aero framework utilizes data-driven methods based on a curated database of physically relevant airfoils. 
Specific tools include: 
 
   -  principal geodesic analysis over normal coordinate neighborhoods of matrix manifolds; 
   -  a variety of data-regularized deformations to nominal 2D airfoil shapes; 
   -  Riemannian interpolation connecting a sequence of airfoil cross-sections to build 3D blades from 2D data; 
   -  consistent perturbations over the span of interpolated 3D blades based on dominant modes from the data-driven analysis. 

Organization
------------

Documentation is currently organized into three main categories:

   * :ref:`How to Guides`: User guides covering basic topics and use cases for the G2Aero software
   * :ref:`Explanation`: Information and research sources for basic concepts used in G2Aero
   * :ref:`Technical Reference`: Programming details on the G2Aero API and functions

New users may find it helpful to review the :ref:`Getting Started` materials first.

Citations     
---------
If you use this software in your research or publications, please use the following BibTeX citations::

   
    @article{Doronina_JOSS_2023, 
      author = {Olga A. Doronina and Zachary J. Grey and Andrew Glaws}, 
      title = {G2Aero: A Python package for separable shape tensors}, 
      journal = {Journal of Open Source Software}, 
      publisher = {The Open Journal}, 
      year = {2023}, 
      volume = {8}, 
      number = {89}, 
      pages = {5408},
      doi = {10.21105/joss.05408}, 
      url = {https://doi.org/10.21105/joss.05408}, 
    }

    @article{GreyJCDE2023,
      author = {Grey, Zachary J and Doronina, Olga A and Glaws, Andrew},
      title = "{Separable shape tensors for aerodynamic design}",
      journal = {Journal of Computational Design and Engineering},
      volume = {10},
      number = {1},
      pages = {468-487},
      year = {2023},
      month = {01},
      doi = {10.1093/jcde/qwac140},
      url = {https://doi.org/10.1093/jcde/qwac140},
   }

   @inproceedings{grassmannian2022,
      title={Grassmannian Shape Representations for Aerodynamic Applications},
      author={Olga Doronina and Zachary Grey and Andrew Glaws},
      booktitle={AAAI 2022 Workshop on AI for Design and Manufacturing (ADAM)},
      year={2022},
      url={https://openreview.net/forum?id=1RRU6ud9YC}
   }

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   how_to_guides/index
   explanation/index
   technical_reference/index
   community


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
