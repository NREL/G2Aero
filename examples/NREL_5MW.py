import os
import numpy as np
from g2aero import yaml_info
from g2aero.Grassmann_interpolation import GrassmannInterpolator
from g2aero.transform import TransformBlade
from g2aero.transform import global_blade_coordinates
from g2aero import geometry_gmsh
from time import time
import datetime
# from plot_animation import plot_3d_blade

def main():
    shapes_filename = os.path.join(os.getcwd(), "../data", 'blades_yamls', "nrel5mw_ofpolars.yaml")

    t_start = time()
    Blade = yaml_info.YamlInfo(shapes_filename, n_landmarks=321)
    eta_nominal = Blade.eta_nominal
    xy_nominal = Blade.xy_landmarks

    M_yaml = Blade.M_yaml_interpolator
    b_yaml = Blade.b_yaml_interpolator
    b_pitch = Blade.pitch_axis

    t1 = time()

    t2 = time()
    print('Grassmann')
    Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
    M = Grassmann.interpolator_M
    b = Grassmann.interpolator_b
    eta_span = np.linspace(0, 1, 200)
    _, gr_crosssections = Grassmann(eta_span, grassmann=True)
    t3 = time()

    print('Inverse Transform')
    Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
    xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)

    # xyz_nominal = np.empty((xy_nominal.shape[0], xy_nominal.shape[1], 3))
    # for i, xy in enumerate(Blade.xy_landmarks):
    #     xyz_nominal[i] = np.hstack((xy, Blade.z_max*eta_nominal[i]*np.ones((xy_nominal.shape[1], 1))))
    # plot_3d_blade(xyz_nominal)

    print("creating CAD model")
    t4 = time()
    xyz_global = global_blade_coordinates(xyz_local)
    # to create stp file and unstructured grid
    geometry_gmsh.blade_CAD_geometry(xyz_global, 'nrel5mw_ofpolars', msh=True)
    t5 = time()

    print(f'Time to make nominal cross sections from .yaml file: {datetime.timedelta(seconds=(t1 - t_start))}')
    print(f'Time to generate cross sections by Grassmann class: {datetime.timedelta(seconds=(t3 - t2))}')
    print(f'Time to generate Bsplines and mesh by gmsh: {datetime.timedelta(seconds=(t5 - t4))}')
    print(f'Time total: {datetime.timedelta(seconds=(t5 - t_start))}')


if __name__ == '__main__':
    main()
