import os
import numpy as np
from unittest import TestCase
from g2aero.yaml_info import YamlInfo
from g2aero.transform import TransformBlade, global_blade_coordinates
from g2aero.Grassmann_interpolation import GrassmannInterpolator


class Test(TestCase):
    def test_NREL_5MW(self):
        shapes_filename = os.path.join(os.getcwd(), "data", 'blades_yamls', "nrel5mw_ofpolars.yaml")

        Blade = YamlInfo.init_from_yaml(shapes_filename, n_landmarks=321)
        eta_nominal = Blade.eta_nominal
        xy_nominal = Blade.xy_landmarks

        M_yaml = Blade.M_yaml_interpolator
        b_yaml = Blade.b_yaml_interpolator
        b_pitch = Blade.pitch_axis

        Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
        M = Grassmann.interpolator_M
        b = Grassmann.interpolator_b
        eta_span = np.linspace(0, 1, 100)
        _, gr_crosssections = Grassmann(eta_span, grassmann=True)

        # Inverse Transform
        Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
        xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)

        # creating CAD model
        xyz_global = global_blade_coordinates(xyz_local)
        # # to create stp file and unstructured grid
        # blade_CAD_geometry(xyz_global, 'nrel5mw_ofpolars', msh=True)

    def test_IEA_15(self):
        shapes_filename = os.path.join(os.getcwd(), "data", 'blades_yamls', "IEA-15-240-RWT.yaml")

        Blade = YamlInfo.init_from_yaml(shapes_filename, n_landmarks=401)
        eta_nominal = Blade.eta_nominal
        xy_nominal = Blade.xy_landmarks

        M_yaml = Blade.M_yaml_interpolator
        b_yaml = Blade.b_yaml_interpolator
        b_pitch = Blade.pitch_axis

        Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
        M = Grassmann.interpolator_M
        b = Grassmann.interpolator_b
        eta_span = np.linspace(0, 1, 100)
        _, gr_crosssections = Grassmann(eta_span, grassmann=True)

        # Inverse Transform
        Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
        xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)

        # creating CAD model
        xyz_global = global_blade_coordinates(xyz_local)

    def test_IEA_10(self):
        shapes_filename = os.path.join(os.getcwd(), "data", 'blades_yamls', "IEA-10-198-RWT.yaml")

        Blade = YamlInfo.init_from_yaml(shapes_filename, n_landmarks=200)
        eta_nominal = Blade.eta_nominal
        xy_nominal = Blade.xy_landmarks

        M_yaml = Blade.M_yaml_interpolator
        b_yaml = Blade.b_yaml_interpolator
        b_pitch = Blade.pitch_axis

        Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
        M = Grassmann.interpolator_M
        b = Grassmann.interpolator_b
        eta_span = np.linspace(0, 1, 100)
        _, gr_crosssections = Grassmann(eta_span, grassmann=True)

        # Inverse Transform
        Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
        xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)

        # creating CAD model
        xyz_global = global_blade_coordinates(xyz_local)

    def test_IEA_3_4(self):
        shapes_filename = os.path.join(os.getcwd(), "data", 'blades_yamls', "IEA-3.4-130-RWT.yaml")

        Blade = YamlInfo.init_from_yaml(shapes_filename, n_landmarks=500)
        eta_nominal = Blade.eta_nominal
        xy_nominal = Blade.xy_landmarks

        M_yaml = Blade.M_yaml_interpolator
        b_yaml = Blade.b_yaml_interpolator
        b_pitch = Blade.pitch_axis

        Grassmann = GrassmannInterpolator(eta_nominal, xy_nominal)
        M = Grassmann.interpolator_M
        b = Grassmann.interpolator_b
        eta_span = np.linspace(0, 1, 100)
        _, gr_crosssections = Grassmann(eta_span, grassmann=True)

        # Inverse Transform
        Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
        xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)

        # creating CAD model
        xyz_global = global_blade_coordinates(xyz_local)

    def test_number_crosssections(self):
        shapes_filename = os.path.join(os.getcwd(), "data", 'blades_yamls', "nrel5mw_ofpolars.yaml")
        Blade = YamlInfo.init_from_yaml(shapes_filename, n_landmarks=401)
        M_yaml = Blade.M_yaml_interpolator
        b_yaml = Blade.b_yaml_interpolator
        b_pitch = Blade.pitch_axis
        Grassmann = GrassmannInterpolator(Blade.eta_nominal, Blade.xy_landmarks)
        for n in [3, 10, 10000]:
            eta_span = np.linspace(0, 1, n)
            _, gr_crosssections = Grassmann(eta_span, grassmann=True)
            M = Grassmann.interpolator_M
            b = Grassmann.interpolator_b
            Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
            xyz_local = Transform.grassmann_to_phys(gr_crosssections, eta_span)

    def test_number_landmarks(self):
        shapes_filename = os.path.join(os.getcwd(), "data", 'blades_yamls', "nrel5mw_ofpolars.yaml")

        for n in [100, 501, 10000]:
            for method in ['cst', 'planar', 'polar']:
                Blade = YamlInfo.init_from_yaml(shapes_filename, n_landmarks=n, landmark_method=method)
                M_yaml = Blade.M_yaml_interpolator
                b_yaml = Blade.b_yaml_interpolator
                b_pitch = Blade.pitch_axis
                Grassmann = GrassmannInterpolator(Blade.eta_nominal, Blade.xy_landmarks)
                M = Grassmann.interpolator_M
                b = Grassmann.interpolator_b
                _, gr_crosssections = Grassmann(np.linspace(0, 1, 100), grassmann=True)
                Transform = TransformBlade(M_yaml, b_yaml, b_pitch, M, b)
                xyz_local = Transform.grassmann_to_phys(gr_crosssections, np.linspace(0, 1, 100))