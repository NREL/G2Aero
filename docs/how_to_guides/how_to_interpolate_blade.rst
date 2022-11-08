.. default-role:: math

Grassmannian Interpolation
======================================

**Input to create interpolator**:

    * 2D shapes (cross sections)
    * normalized locations of these shapes along the 3rd direction

**Requirements**:

    * 2D shapes parallel to each other 
    * Equal number of landmarks (points defining a shape in 2D)
    * Locations along the 3rd direction are mormalized from 0 to 1

Interpolation for a blade defined as .yaml file
------------------------------------------------

Usually, a wind turbine blade definition is provided by a .yaml file which contains cross-sectional airfoils (normalized by chord size), 
their location along the blade span, and profiles of pitch axis (axis of twist), pitch angle (twist), scaling, and shift.

Our routine for blade interpolation consists of the following steps:

  1. Read the blade definition from the .yaml file and reparametrize given cross-section airfoils so they have an equal number of landmarks.
  2. Interpolate shapes between given cross-sections with the unit chord.
  3. Apply affine transformation to the interpolated shapes to scale, rotate, shift and bend (out-of-plane rotation) the blade according to the profiles provided in .yaml file.

This example also can be found as a script in ``G2Aero/examples/blade_interpolation.py``

To demonstrate this example we need to import some modules

    >>> import os
    >>> import numpy as np
    >>> from g2aero.yaml_info import YamlInfo
    >>> from g2aero.Grassmann_interpolation import GrassmannInterpolator
    >>> from g2aero.transform import TransformBlade, global_blade_coordinates

Reading blade definition from .yaml file
```````````````````````````````````````````

We first create a class object ``Blade`` with information from .yaml file and 
save ``Blade.xy_landmarks`` (2D cross-sectional airfoils reparametrized to have the same number of landmarks)
and ``Blade.eta_nominal`` (locations from 0 to 1 of these cross sections along the normalized blade span) to use as input for the interpolator.
Note that the number of landmarks is defined by the user and set to ``n_landmarks=401`` in this example.

    >>> shapes_filename = os.path.join(os.getcwd(), "../../data/blades_yamls/IEA-15-240-RWT.yaml")
    >>> Blade = YamlInfo(shapes_filename, n_landmarks=401) 
    >>> xy_nominal = Blade.xy_landmarks 
    >>> eta_nominal = Blade.eta_nominal 

Interpolation
`````````````

Then we create the interpolator ``GrInterp`` with given 2D cross-sections and their spanwise location as input parameters.

    >>> GrInterp = GrassmannInterpolator(eta_nominal, xy_nominal)

Now we need to define an array of spanwise locations where we want to get new interpolated cross-sections. We can provide any desired locations, e.g. 500 locations uniform along the blade span

    >>> eta_span = np.linspace(0, 1, 100)

or generate locations using the interpolator method. It distributes locations according to the Grassmann distance between given shapes. 
Note that this method also has arguments ``n_hub``, ``n_tip``, ``n_end``, which can help specify locations near the hub and near the tip (see Technical Reference for method details)

    >>> eta_span = GrInterp.sample_eta(n_samples=100)
    
Next, we pass the array of desired locations to the interpolator to get interpolated shapes. ``phys_crosssections`` contains 2D interpolated shapes parallel to each other and with the unit chord.
    
    >>> phys_crosssections, gr_crosssections = GrInterp(eta_span, grassmann=True)

Apply affine transformation
```````````````````````````

Finally, we apply affine transformation to the interpolated shapes to scale, rotate, shift and bend (out-of-plane rotation) the blade 
according to the profiles provided in .yaml file. 

>>> M_yaml = Blade.M_yaml_interpolator
>>> b_yaml = Blade.b_yaml_interpolator
>>> b_pitch = Blade.pitch_axis
>>> M = GrInterp.interpolator_M
>>> b = GrInterp.interpolator_b



>>> Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
>>> xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)
>>> xyz_global = global_blade_coordinates(xyz_local)


.. raw:: html
   :file: files/interpolated_blade.html