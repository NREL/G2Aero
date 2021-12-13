import os
import numpy as np
from g2aero.yaml_info import YamlInfo
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.transform import TransformBlade, global_blade_coordinates
from g2aero.geometry_gmsh import blade_CAD_geometry, write_geofile
from time import time
import datetime


def main():
    shapes_filename = os.path.join(os.getcwd(), "../data", 'blades_yamls', "IEA-15-240-RWT.yaml")

    t_start = time()
    Blade = YamlInfo(shapes_filename, n_landmarks=401)
    # Blade.make_straight_blade()
    eta_nominal = Blade.eta_nominal
    xy_nominal = Blade.xy_nominal

    M_yaml = Blade.M_yaml_interpolator
    b_yaml = Blade.b_yaml_interpolator
    b_pitch = Blade.pitch_axis
    t1 = time()

    t2 = time()
    print('Grassmann')
    Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
    M = Grassmann.interpolator_M
    b = Grassmann.interpolator_b
    eta_span = np.linspace(0, 1, 100)
    print(len(eta_span))
    _, gr_crosssections = Grassmann(eta_span, grassmann=True)
    t3 = time()

    print('Inverse Transform')
    Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
    xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)
    print("creating CAD model")
    t4 = time()
    xyz_global = global_blade_coordinates(xyz_local)
    # np.savez('xyz_local.npz', xyz=xyz_local)

    # to create stp file and unstructured grid
    blade_CAD_geometry(xyz_global, 'IEA-15-240-RWT', msh=True)

    # to create structured grid
    # write_geofile(xyz_global, 'nrel5mw_ofpolars', n_spanwise=300, n_te=3, n_cross_half=100)
    t5 = time()

    print(f'Time to make nominal cross sections from .yaml file: {datetime.timedelta(seconds=(t1 - t_start))}')
    print(f'Time to generate cross sections by Grassmann class: {datetime.timedelta(seconds=(t3 - t2))}')
    print(f'Time to generate Bsplines and mesh by gmsh: {datetime.timedelta(seconds=(t5 - t4))}')
    print(f'Time total: {datetime.timedelta(seconds=(t5 - t_start))}')


if __name__ == '__main__':
    main()
