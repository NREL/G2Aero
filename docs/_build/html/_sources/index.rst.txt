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
* :ref:`Technical Reference`: Programming details on the G2Aero API and functions
* :ref:`Explanation`: Information and research sources for basic concepts used in G2Aero


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   how_to_guides/index
   technical_reference/index
   explanation/index
   usage/installation
   community


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
