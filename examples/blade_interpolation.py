import os
import numpy as np
from g2aero.yaml_info import YamlInfo
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.transform import TransformBlade, global_blade_coordinates
from g2aero import geometry_gmsh
from time import time
import datetime

examples_path = os.path.dirname(__file__)
root_path = os.path.abspath(os.path.join(examples_path, os.pardir))

def main():

    blade_filename = 'IEA-15-240-RWT.yaml'
    # blade_filename = 'nrel5mw_ofpolars.yaml'

    shapes_path = os.path.join(root_path, "data", 'blades_yamls', blade_filename)
    n_landmarks = 1001
    n_cross_sections = 200
    print(f'Number of landmarks: {n_landmarks}')
    print(f'Number of interpolated cross sections: {n_cross_sections}')
    
    t_start = time()
    
    Blade = YamlInfo.init_from_yaml(shapes_path, n_landmarks=n_landmarks, add_gap=0.002)
    eta_nominal = Blade.eta_nominal
    xy_nominal = Blade.xy_nominal

    M_yaml = Blade.M_yaml_interpolator
    b_yaml = Blade.b_yaml_interpolator
    b_pitch = Blade.pitch_axis
    
    t1 = time()

    print('Grassmann')
    Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
    M = Grassmann.interpolator_M
    b = Grassmann.interpolator_b
    eta_span = np.linspace(0, 1, n_cross_sections)
    
    _, gr_crosssections = Grassmann(eta_span, grassmann=True)
    
    t2 = time()

    print('Inverse Transform')
    Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
    xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)
    xyz_global = global_blade_coordinates(xyz_local)
    np.savez('xyz_blade.npz', xyz=xyz_global)
    
    t3 = time()
    
    # print("creating CAD model")
    # TODO: check gmsh functionality (need to have at least a minimum gap to work)
    # # to create stp file and unstructured grid
    geometry_gmsh.blade_CAD_geometry(xyz_global, blade_filename[:-5], msh=True)
    # to create structured grid
    geometry_gmsh.write_geofile(xyz_global, blade_filename[:-5], n_spanwise=300, n_te=3, n_cross_half=100)
    
    t4 = time()

    print(f'Time to make nominal cross sections from .yaml file: {datetime.timedelta(seconds=(t1 - t_start))}')
    print(f'Time to generate cross sections by Grassmann class: {datetime.timedelta(seconds=(t2 - t1))}')
    print(f'Time to transform generated cross sections to final physical coordinates: {datetime.timedelta(seconds=(t3 - t2))}')
    print(f'Time to generate Bsplines and mesh by gmsh: {datetime.timedelta(seconds=(t4 - t3))}')
    print(f'Time total: {datetime.timedelta(seconds=(t4 - t_start))}')


if __name__ == '__main__':
    main()
