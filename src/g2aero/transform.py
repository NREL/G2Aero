import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator


class TransformBlade:

    def __init__(self, M_yaml, b_yaml, b_pitch, M, b):
        # Pchip interpolators
        self.M_yaml = M_yaml
        self.b_yaml = b_yaml
        self.M = M
        self.b = b
        self.b_pitch = b_pitch


    def make_R_out_interpolator(self, eta):
        """ Airfoils shapes have to be rotated out of xy plane to be normal 
        to y component (in local coord) of ref axis (elastic axis).

        :param eta: locations along the blade span (need to define the range)
        :return: rotation interpolator
        """
        eta = np.linspace(eta[0], eta[-1], 200)
        grad = np.gradient(self.b_yaml(eta)[:, 1], self.b_yaml(eta)[:, 2])
        angles = np.arctan(grad)
        rotations = Rotation.from_euler('x', -angles, degrees=False)
        return Slerp(eta, rotations)

    def grassmann_to_nominal(self, xy_gr, eta):
        xy_nominal = np.empty_like(xy_gr)
        for i, (eta, xy) in enumerate(zip(eta, xy_gr)):
            xy_nominal[i] = xy @ self.M(eta).T + self.b(eta)
        return xy_nominal

    def grassmann_to_phys(self, xy_gr, eta):
        n_shapes, n_landmarks, _ = xy_gr.shape
        xyz_phys = np.empty((n_shapes, n_landmarks, 3))
        self.R_out = self.make_R_out_interpolator(eta)

        print(self.R_out(eta).as_matrix)

        for i, (eta, xy) in enumerate(zip(eta, xy_gr)):
            M_total, b_total = self.calc_M_b_total(eta)
            M_total_xy = xy @ M_total.T
            xyz = np.hstack((M_total_xy, np.zeros((n_landmarks, 1))))
            xyz_phys[i] = self.R_out(eta).apply(xyz) + b_total

        return xyz_phys

    def calc_M_b_total(self, eta):
        M_yaml_3d = np.eye(3)
        M_yaml, b_yaml = self.M_yaml(eta), self.b_yaml(eta)
        M_yaml_3d[:2, :2] = M_yaml
        M, b = self.M(eta), self.b(eta)
        b = np.concatenate((b, [0]))
        R_out = self.R_out(eta)

        b_total = R_out.as_matrix() @ M_yaml_3d @ b + b_yaml
        M_total = self.M_yaml(eta) @ self.M(eta)
        return M_total, b_total

    def update_M_yaml_interpolator(self, x_twist, twist, x_chord, chordx, chordy):

        theta = PchipInterpolator(x_twist, twist)
        M_yaml = np.empty((len(x_chord), 2, 2))
        for i, eta in enumerate(x_chord):
            c, s = np.cos(theta(eta)), np.sin(theta(eta))
            R_twist = np.array([[c, -s], [s, c]])
            M_chord = np.diag([chordx[i], chordy[i]])
            M_yaml[i] = R_twist @ M_chord
        self.M_yaml = PchipInterpolator(x_chord, M_yaml)


    # def calc_rotation_interpolator(self):
    #     """ Airfoils shapes have to be rotated out of xy plane to be normal
    #     to y component (in local coord) of ref axis (elastic axis).
    #
    #     :return: rotation interpolator
    #     """
    #     grad = np.empty_like(self.shift_values)
    #     for i, shift_i in enumerate(self.shift_values.T):
    #         grad[:, i] = np.gradient(shift_i, self.shift_grid)
    #     grad /= np.linalg.norm(grad, axis=1, keepdims=True)
    #     z_axis = np.array([0, 0, 1])
    #     quaternions = np.empty((len(self.shift_values), 4))
    #     for i, gr in enumerate(grad):
    #         c = np.dot(z_axis, gr)
    #         theta = np.arccos(c)
    #         quaternions[i] = [np.sin(theta/2), np.sin(theta/2), np.sin(theta/2), np.cos(theta/2)]
    #     rotations = Rotation.from_quat(quaternions)
    #
    #     return Slerp(self.shift_grid, rotations)


def transform_for_BEM(eta, xy, twist, scalex, scaley, pitch):
    xy_scaled = xy.copy()
    xy_scaled[:, 0] -= pitch(eta)
    theta = twist(eta)
    c, s = np.cos(theta), np.sin(theta)
    R_twist = np.array([[c, -s], [s, c]])
    M_chord = np.diag([scalex(eta), scaley(eta)])
    M = R_twist @ M_chord
    return xy_scaled@M.T


def transform_blade_for_BEM(eta_span, shapes, twist, scalex, scaley, pitch):
    """

    :param eta_span: locations of cross sections on (0, 1)
    :param shapes: grassmann shapes
    :param twist: twist interpolator
    :param scalex: chord interpolator (should be the same as scaley)
    :param scaley: chord interpolator
    :param pitch: pitch interpolator
    :return:
    """
    shapes_transformed = np.empty_like(shapes)
    for i, (eta, xy) in enumerate(zip(eta_span, shapes)):
        shapes_transformed[i] = transform_for_BEM(eta, xy, twist, scalex, scaley, pitch)
    return shapes_transformed


def global_blade_coordinates(xyz_local):

    n_shapes, n_landmarks, _ = xyz_local.shape
    xyz_global = np.empty((n_shapes, n_landmarks, 3))
    for i, xyz in enumerate(xyz_local):
        xyz_global[i] = np.c_[xyz[:, 1], -xyz[:, 0], xyz[:, 2]]
    return xyz_global