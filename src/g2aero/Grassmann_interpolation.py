import numpy as np
from scipy.interpolate import PchipInterpolator
from g2aero.Grassmann import exp, log, distance, landmark_affine_transform


class GrassmannInterpolator:

    def __init__(self, eta, xy):
        """

        :param eta: list of shape locations eta (normalized from 0 to 1)
        :param xy: (n_shapes, n_landmarks, 2) array with 2D shape coordinates in physical space (nominal shapes)
        """
        # sort based on increasing span
        self.xy_nominal = np.array(xy)[np.argsort(eta)]
        self.eta_nominal = np.sort(eta)
        self.n_shapes, self.n_landmarks, _ = self.xy_nominal.shape

        # LA transformation
        self.xy_grassmann, self.M, self.b = landmark_affine_transform(self.xy_nominal)
        self.dist_grassmann = self.calc_grassmann_distance(self.xy_grassmann)

        self.t_nominal = np.cumsum(self.dist_grassmann)
        self.t_nominal /= np.max(self.t_nominal)        # normalized to [0, 1]

        # renormalize eta to have nonduplicating shapes between 0 and 1
        self.eta_shift, self.eta_scale = self.renormalize_eta(self.dist_grassmann)
        self.eta_nominal_scaled = self.eta_scaled(self.eta_nominal)

        self.interpolator_cdf_01 = self.make_interpolator_cdf()

        self.log_map = self.calc_log_mapping(self.xy_grassmann)

        self.interpolator_M = PchipInterpolator(self.eta_nominal, self.M, axis=0)
        self.interpolator_b = PchipInterpolator(self.eta_nominal, self.b, axis=0)

    def eta_scaled(self, eta):
        """Renormalized eta, such that duplicated shapes are outside of [0, 1] interval.

        :param eta: given value of eta (single value of array)
        :return: Renormalized eta (single value of array)
        """
        return (eta - self.eta_shift) / self.eta_scale

    def eta_scaled_inv(self, eta):
        return eta * self.eta_scale + self.eta_shift

    def renormalize_eta(self, dist):
        """Renormalized eta, such that duplicated shapes (zero distance) in the beginning and in the end
        are outside of [0, 1] interval.

        :param dist: (n_shapes, ) array of calculated distances between shapes (dist[0] always equal 0)
        :return: eta_shift, eta_scale for renormalization
        """
        # check for duplicates in the beginning and in the end
        ind_no_duplicates = np.arange(len(dist))
        while dist[1] == 0:
            dist = np.delete(dist, 0)
            ind_no_duplicates = np.delete(ind_no_duplicates, 0)
        while dist[-1] == 0:
            dist = np.delete(dist, -1)
            ind_no_duplicates = np.delete(ind_no_duplicates, -1)
        # renormalize
        eta = self.eta_nominal[ind_no_duplicates]
        eta_shift = np.min(eta)
        eta_scale = np.max(eta) - np.min(eta)
        return eta_shift, eta_scale

    def make_interpolator_cdf(self):

        # PCHIP 1-D monotonic cubic interpolatio
        interpolator = PchipInterpolator(self.eta_nominal_scaled, self.t_nominal)
        eta_span = np.linspace(self.eta_nominal_scaled[0], self.eta_nominal_scaled[-1], 100000)
        t_span = interpolator(eta_span)
        interpolator_cdf = PchipInterpolator(eta_span, t_span)
        return interpolator_cdf

    def interpolator_cdf(self, eta_scaled):
        if np.isscalar(eta_scaled):
            if 0 < eta_scaled < 1:
                return self.interpolator_cdf_01(eta_scaled)
            elif eta_scaled <= 0:
                return 0.0
            else:
                return 1.0
        else:
            t = np.zeros_like(eta_scaled)
            ind_01 = np.logical_and(0 < eta_scaled, eta_scaled < 1)
            t[ind_01] = self.interpolator_cdf_01(eta_scaled[ind_01])
            t[eta_scaled >= 1] = 1.0
            return t

    @staticmethod
    def calc_grassmann_distance(xy_grassmann):
        """Calculate distance between ordered elements on Grassmann.

        :param xy_grassmann: (n_shapes, n_landmarks, 2) array of elemments on Grassmann
        :return: (n_shapes, ) array of distances on Grassmann (dist[0] always equal 0)
        """
        dist = np.zeros(len(xy_grassmann))
        for i in range(1, len(xy_grassmann)):
            dist[i] = distance(xy_grassmann[i], xy_grassmann[i - 1])
        return dist

    @staticmethod
    def calc_log_mapping(xy_grassmann):
        """Calculate log mapping (the tangent direction from Xi to Xi+1)
        for every given element Xi in ordered list of elements.

        :param xy_grassmann: (n_shapes, n_landmarks, 2) array of elements on Grassmann
        :return: (n_shapes-1, ) array of directions (log mapping)
        """
        n_shapes, n_landmarks, dim = xy_grassmann.shape
        log_map = np.empty((n_shapes - 1, n_landmarks, dim))
        for i in range(n_shapes-1):
            log_map[i] = log(xy_grassmann[i], xy_grassmann[i+1])  # compute direction
        return log_map

    def sample_eta(self, n_samples, n_hub=10, n_tip=None, n_end=25):
        """Sample eta, such that t is uniformly distributed and filling
        uniformly distributed eta between duplicated shapes.

        :param n_samples: number of samples uniformly distributed over Grassmannian
        :param n_hub: number of samples uniform over eta between duplicated shapes on the left
        :param n_tip: number of samples uniform over eta between duplicated shapes on the right, not including tip end
        :param n_end: number of samples uniform over eta for the tip end
        :return: array of eta samples
        """
        # make inverse interpolator
        eta_scaled_01_tmp = np.linspace(0, 1, 10000)
        t_tmp = self.interpolator_cdf(eta_scaled_01_tmp)
        interpolator_inv_01 = PchipInterpolator(t_tmp, eta_scaled_01_tmp)

        if n_end == 0:
            tip_end = 0.0
        else:
            tip_end = self.eta_nominal_scaled[-1] * 0.02
        if not n_tip:
            n_tip = 0.2*n_samples
        t_span = np.linspace(0, 1, n_samples + 2)
        eta_span = interpolator_inv_01(t_span)
        # add uniform over eta sampling if duplicate shapes
        if self.eta_nominal_scaled[0] < 0.0:
            eta_left = np.linspace(self.eta_nominal_scaled[0], 0, int(n_hub), endpoint=False)
            eta_span = np.hstack((eta_left, eta_span))
        if self.eta_nominal_scaled[-1] > 1.0:
            eta_right = np.linspace(1, self.eta_nominal_scaled[-1] - tip_end, int(n_tip), endpoint=False)[1:]
            eta_span = np.hstack((eta_span, eta_right))

        # refine tip end
        if tip_end != 0.0:
            eta_end = np.linspace(self.eta_nominal_scaled[-1] - tip_end, self.eta_nominal_scaled[-1], int(n_end)+1)[1:]
            eta_span = np.hstack((eta_span, eta_end))
        return self.eta_scaled_inv(eta_span)

    def shapes_perturbation(self, new_shapes, ind):
        """ Substitute nominal shapes with perturbed ones.

        :param new_shapes: perturbed grassmann shapes
        :param ind: indices of shapes to substitute
        """
        self.xy_grassmann[ind] = new_shapes
        self.dist_grassmann = self.calc_grassmann_distance(self.xy_grassmann)

        self.t_nominal = np.cumsum(self.dist_grassmann)
        self.t_nominal /= np.max(self.t_nominal)  # normalized to [0, 1]

        self.interpolator_cdf_01 = self.make_interpolator_cdf()
        self.log_map = self.calc_log_mapping(self.xy_grassmann)

    def __call__(self, eta, grassmann=False):
        """Interpolate 2D shapes at a given spanwise locations.

        :param eta: physical normalized spanwise locations (assume eta is in [0,1])
        :param grassmann: if True, return shape in grassmann space as well (Landmark-Affine standardized)
        :return: array of (n, 2) arrays 2D shape coordinates
        """
        eta = np.asarray(eta)
        scalar_input = False
        if eta.ndim == 0:
            eta = eta[np.newaxis]  # Makes x 1D
            scalar_input = True
        shapes_phys = np.empty((eta.size, self.n_landmarks, 2))
        if grassmann:
            shapes_gr = np.empty_like(shapes_phys)
        for i, et in enumerate(eta):
            # locate given eta according the nominal etas
            eta_sc = self.eta_scaled(et)
            t = self.interpolator_cdf(eta_sc)
            ind = np.where(self.t_nominal <= t)[-1][-1]
            t_start = self.t_nominal[ind]
            t_end = self.t_nominal[min(len(self.t_nominal) - 1, ind + 1)]   # in case t_start is end point
            # Grassmannian
            if t == 0 or t == 1:
                geodesic_grassmann = self.xy_grassmann[ind]
            else:
                t_norm = (t - t_start) / (t_end - t_start)
                geodesic_grassmann = exp(t_norm, self.xy_grassmann[ind], self.log_map[ind])

            b = self.interpolator_b(et)
            M = self.interpolator_M(et)
            shapes_phys[i] = geodesic_grassmann @ M.T + b
            if grassmann:
                shapes_gr[i] = geodesic_grassmann

        if scalar_input:
            shapes_phys, shapes_gr = shapes_phys.squeeze(axis=0), geodesic_grassmann

        if grassmann:
            return shapes_phys, shapes_gr
        else:
            return shapes_phys
