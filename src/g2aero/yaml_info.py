import yaml
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator, interp1d
from .reparametrization import get_landmarks


class YamlInfo:

    def __init__(self, yaml_filename, n_landmarks):
        airfoils_dict, airfoils_nominal_list, hub_d = self.read_yamlfile(yaml_filename)

        self.eta_nominal = np.array(airfoils_dict['airfoil_position']['grid'])
        self.labels_nominal = airfoils_dict['airfoil_position']['labels']

        self.xy_fromfile = self.get_xy_coordinates(airfoils_nominal_list, self.labels_nominal)
        self.n_landmarks = n_landmarks
        self.xy_landmarks = np.empty((len(self.labels_nominal), n_landmarks, 2))
        for i, xy in enumerate(self.xy_fromfile):
            # self.xy_landmarks[i] = get_landmarks(xy, n_landmarks=n_landmarks, add_gap=0.002, name=self.labels_nominal[i])
            self.xy_landmarks[i] = get_landmarks(xy, n_landmarks=n_landmarks, add_gap=0.002)

        # scaling of the nominal shape (same in x and y direction)
        self.chord_values = np.array(airfoils_dict['chord']['values'])
        self.chord_grid = np.array(airfoils_dict['chord']['grid'])
        self.chord = PchipInterpolator(self.chord_grid, self.chord_values)

        # angle (in rad) of 2d rotation (xy)
        self.twist_values = np.array(airfoils_dict['twist']['values'])
        self.twist_grid = np.array(airfoils_dict['twist']['grid'])
        self.twist = PchipInterpolator(self.twist_grid, self.twist_values)

        self.M_yaml_interpolator = self.make_M_yaml_interpolator()

        # in local coordinates
        # axis of rotation (shift in x direction only)
        self.pitch_axis_values = np.array(airfoils_dict['pitch_axis']['values'])
        self.pitch_axis_grid = np.array(airfoils_dict['pitch_axis']['grid'])
        # self.pitch_axis_values = np.vstack((self.pitch_axis_values, np.zeros_like(self.pitch_axis_grid))).T
        self.pitch_axis = PchipInterpolator(self.pitch_axis_grid, self.pitch_axis_values)
        self.xy_nominal = self.shift_to_pitch_axis(self.eta_nominal)

        # in yaml file defined in global coordinates
        # shift of the shape after twist rotation (can happen in x and y direction). Bending blade span
        # z axis is location along the blade span
        self.make_shift_interpolator(airfoils_dict, hub_d)
        self.b_yaml_interpolator = self.shift

        # rotation out of xy plane, so shapes are normal to the ref_axis (elastic axis)
        # self.elastic_rotations = self.calc_rotation_interpolator()
        self.angle4elastic_rotation = self.calc_angle_interpolator()

        self.xy_transformed = self.get_physical_xy(self.xy_nominal, self.eta_nominal)

    @staticmethod
    def read_yamlfile(filename):
        with open(filename, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.Loader)
        return yaml_dict['components']['blade']['outer_shape_bem'], yaml_dict['airfoils'], yaml_dict['components']['hub']['diameter']

    @staticmethod
    def get_xy_coordinates(airfoils_list, nominal_labels):
        xy_nominal = []
        names_dict = {}
        for name in airfoils_list:
            names_dict[name['name']] = np.vstack((name['coordinates']['x'], name['coordinates']['y'])).T
        for i, label in enumerate(nominal_labels):
            xy_nominal.append(names_dict[label])
        return xy_nominal

    def shift_to_pitch_axis(self, etas):
        xy_nominal = self.xy_landmarks.copy()
        xy_nominal[:, :, 0] -= self.pitch_axis(etas).reshape(-1, 1)
        return xy_nominal

    def make_M_yaml_interpolator(self):
        # evaluate at chord grid, because chord has bigger gradient
        M_yaml = np.empty((len(self.chord_grid), 2, 2))
        for i, eta in enumerate(self.chord_grid):
            c, s = np.cos(self.twist(eta)), np.sin(self.twist(eta))
            R_twist = np.array([[c, -s], [s, c]])
            M_chord = np.diag(np.ones(2) * self.chord(eta))
            M_yaml[i] = R_twist @ M_chord
        return PchipInterpolator(self.chord_grid, M_yaml)

    def make_shift_interpolator(self, airfoils_dict, hub_d):
        # in yaml file defined in global coordinates
        # shift of the shape after twist rotation (can happen in x and y direction). Bending blade span
        # z axis is location along the blade span
        ref_axis_values = dict()
        ref_axis_grid = dict()
        ref_axis = dict()
        for coord in ['x', 'y', 'z']:
            ref_axis_values[coord] = np.array(airfoils_dict['reference_axis'][coord]['values'])
            ref_axis_grid[coord] = np.array(airfoils_dict['reference_axis'][coord]['grid'])
            if coord == 'z':
                ref_axis_values[coord] += hub_d/2
            if len(ref_axis_grid[coord]) > 2:
                ref_axis[coord] = PchipInterpolator(ref_axis_grid[coord], ref_axis_values[coord])
            else:
                ref_axis[coord] = interp1d(ref_axis_grid[coord], ref_axis_values[coord])
        self.shift_grid = ref_axis_grid['z']
        self.z_max = ref_axis_values['z'][-1]
        # in local coordinates x_loc = -y_glob, y_loc = x_glob
        self.shift_values = np.vstack((-ref_axis['y'](self.shift_grid), ref_axis['x'](self.shift_grid), ref_axis_values['z'])).T
        self.shift = PchipInterpolator(self.shift_grid, self.shift_values)

    def calc_angle_interpolator(self):
        """ Create Pchip interpolator of angle of out-of-plane (elastic) rotation in yz plane (in local coord),
        because airfoil shapes have to be rotated out of xy plane to be normal to y component (in local coord)
        of ref axis (elastic axis).
        :return: angle interpolator
        """
        if np.allclose(self.shift_values[:, 1], np.zeros_like(self.shift_values[:, 1])):
            angles = np.zeros_like(self.shift_values[:, 1])
        else:
            grad = np.gradient(self.shift_values[:, 1], self.shift_values[:, 2])
            angles = np.arctan(grad)
        return PchipInterpolator(self.shift_grid, angles)

    def local_transform(self, xy_nominal, eta):
        xy = xy_nominal.copy()

        xyz = np.hstack((xy, np.zeros((self.n_landmarks, 1))))

        theta = self.twist(eta)
        twist_rotation = Rotation.from_euler('z', theta, degrees=False)
        theta2 = self.angle4elastic_rotation(eta)
        outofplane_rotation = Rotation.from_euler('x', -theta2, degrees=False)
        rotation = outofplane_rotation * twist_rotation
        xyz_transformed = rotation.apply((xyz * self.chord(eta))) + self.shift(eta)

        return xyz_transformed

    def get_physical_xy(self, xy_nominal, eta):
        n_shapes, n_landmarks, _ = xy_nominal.shape
        xy_transformed = np.empty((n_shapes, n_landmarks, 3))
        for i, xy in enumerate(xy_nominal):
            xy_transformed[i] = self.local_transform(xy, eta[i])
        return xy_transformed

    def add_eta_nominal(self, eta):
        """
        Adding additional nominal cross section between same shapes
        :param eta: list of additional locations
        :return:
        """
        # making sure they are between two identical shapes
        eta = np.sort(eta)
        eta_ind_left = np.where(self.eta_nominal < eta[0])[0][-1]
        eta_ind_right = np.where(self.eta_nominal > eta[-1])[0][0]
        if self.labels_nominal[eta_ind_left] == self.labels_nominal[eta_ind_right]:
            self.eta_nominal = np.sort(np.append(self.eta_nominal, eta))
            xy_landmarks = np.asarray([self.xy_landmarks[eta_ind_left]]*len(eta))
            self.xy_landmarks = np.concatenate((self.xy_landmarks, xy_landmarks), axis=0)
            self.xy_nominal = self.shift_to_pitch_axis(self.eta_nominal)
        else:
            print("Couldn't add nominal shapes, not all given eta are between identical shapes")

    def make_straight_blade(self):
        self.shift_values[:, 0] = np.zeros_like(self.shift_grid)
        self.shift_values[:, 1] = np.zeros_like(self.shift_grid)
        self.shift = PchipInterpolator(self.shift_grid, self.shift_values)
        self.b_yaml_interpolator = self.shift
        self.angle4elastic_rotation = self.calc_angle_interpolator()
        self.xy_transformed = self.get_physical_xy(self.xy_nominal, self.eta_nominal)


