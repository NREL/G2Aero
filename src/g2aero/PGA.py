import numpy as np
from scipy.interpolate import PchipInterpolator
from g2aero.Grassmann import *
from g2aero.utils import check_selfintersect
import g2aero.SPD as spd

class Dataset:
    def __init__(self, phys_shapes, method='SPD'):
        self.n_shapes = len(phys_shapes)
        if method == 'SPD':
            self.shapes_gr, self.M, self.b = spd.polar_decomposition(phys_shapes)
        elif method == 'LA-transform':
            self.shapes_gr, self.M, self.b = landmark_affine_transform(phys_shapes)


class PGAspace:
    def __init__(self, Vh, M_mean, b_mean, karcher_mean):
        self.n_modes = Vh.shape[0]
        self.n_landmarks, self.ndim = karcher_mean.shape
        self.Vh = Vh
        self.M_mean = M_mean
        self.b_mean = b_mean
        self.karcher_mean = karcher_mean

    @classmethod
    def create_from_dataset(cls, phys_shapes, n_modes=None, method='SPD'):
        if method == 'SPD':
            shapes_gr, M, b = spd.polar_decomposition(phys_shapes)
        elif method == 'LA-transform':
            shapes_gr, M, b = landmark_affine_transform(phys_shapes)
        karcher_mean = Karcher(shapes_gr)

        Vh, S, t = PGA(karcher_mean, shapes_gr, n_coord=n_modes)
        pga_space = cls(Vh, np.mean(M, axis=0), np.mean(b, axis=0), karcher_mean)
        pga_space.S = S

        # for coefficients sampling.
        pga_space.axis_min = np.min(t, axis=0)
        pga_space.axis_max = np.max(t, axis=0)
        r_min = np.abs(np.quantile(t, 0.0, axis=0))
        r_max = np.abs(np.quantile(t, 1.0, axis=0))
        pga_space.radius = np.max(np.array([r_min, r_max]), axis=0)
        return pga_space, t

    def PGA2gr_shape(self, pga_coord, original_shape_gr=None):
        gr_shape = perturb_gr_shape(self.Vh, self.karcher_mean, pga_coord)
        if original_shape_gr is not None:
            R = procrustes(gr_shape, original_shape_gr)
            gr_shape = gr_shape @ R
        return gr_shape

    def PGA2shape(self, pga_coord, M=None, b=None, original_shape_gr=None):
        if M is None:
            M = self.M_mean
        if b is None:
            b = self.b_mean
        gr_shape = perturb_gr_shape(self.Vh, self.karcher_mean, pga_coord)
        # rotate to match with original shape (e.g. for reconstruction error calculation)
        if original_shape_gr is not None:
            R = procrustes(gr_shape, original_shape_gr)
            gr_shape= gr_shape @ R
        return gr_shape @ M + b

    def gr_shapes2PGA(self, shapes_gr):
        return get_PGA_coordinates(shapes_gr, self.karcher_mean, self.Vh.T)
    
    def shapes2PGA(self, shapes, method='SPD', n_modes=None):
        if method == 'SPD':
            shapes_gr, M, b = spd.polar_decomposition(shapes)
        elif method == 'LA-transform':
            shapes_gr, M, b = landmark_affine_transform(shapes)
        t = get_PGA_coordinates(shapes_gr, self.karcher_mean, self.Vh.T)
        return t, M, b

    def sample_coef(self, n_samples=1):
        k = 0
        coef = np.empty((n_samples, self.n_modes))
        while k < n_samples:
            c = np.random.uniform(self.axis_min, self.axis_max)
            # Check that c is inside of ellipse
            if np.sum(c ** 2 / self.radius ** 2) <= 1:
                coef[k] = c
                k += 1
        return coef

    def generate_perturbed_shapes(self, coef=None, n=1):
        """ Generates perturbed shapes.

        If coef are sampled (not given) checks for intersection
        in generated shape and resample if needed.

        :param coef: array of deterministic perturbations (n, n_modes)
        :param n: number of perturbed shapes, if not given calculated from coef.shape[0]
        :return: array of generated perturbed shapes (n, n_landmarks, 2) in physical and 
                 array of grassmann coordinates and 
                 array of PGA coordinates corresponding to perturbations (n, n_modes)
        """
        if coef is None:
            coef_array = self.sample_coef(n)
        else:
            coef_array = coef
        n = len(coef_array)
        gr_samples = np.empty((n, self.n_landmarks, self.ndim))
        phys_samples = np.empty_like(gr_samples)
        for i, c in enumerate(coef_array):
            while True:
                gr_samples[i] = perturb_gr_shape(self.Vh, self.karcher_mean, c)
                if coef is not None or not check_selfintersect(gr_samples[i])       :
                    break
                else:
                    c = self.sample_coef()
                    print(f"WARNING: New shape {i} has intersection! Generating new coef")
                    coef_array[i] = c
            phys_samples[i] = gr_samples[i] @ self.M_mean.T + self.b_mean
        if n == 1:
            return phys_samples.squeeze(axis=0), gr_samples.squeeze(axis=0),  coef_array
        return phys_samples, gr_samples, coef_array

    def generate_perturbed_blade(self, blade, coef=None, n=1):
        """ Generates perturbed blades.

        If coef are sampled (not given) checks for intersection
        in generated shape and resample if needed.

        :param blade: array of grassmann shapes for baseline blade
        :param coef: perturbation coefficients (sampled if not given)
        :param n: number of perturbations (if need to sample)
        :return: array of perturbed blades (in grassmann coordinates) (shape=(n, n_shapes, n_landmarks, 2)) and 
                 array of PGA corresponding to perturbations (n, n_modes)
        """
        n_shapes, n_landmarks, dim = blade.shape

        def get_new_blade(c):
            new_blade = np.empty((n_shapes, n_landmarks, dim))
            vector = (c @ self.Vh).reshape(-1, 2)
            for i, shape in enumerate(blade):
                direction = log(self.karcher_mean, shape)
                new_vector = parallel_translate(self.karcher_mean, direction, vector)
                new_blade[i] = exp(1, shape, new_vector)
            return new_blade

        def intersection_exist_in_blade(blade):
            for i, shape in enumerate(blade):
                if check_selfintersect(shape):
                    print(f"WARNING: New shape {i} has intersection!")
                    return True
            return False

        if coef is None:
            coef_array = self.sample_coef(n)
        else:
            coef_array = np.asarray(coef)
            if len(coef_array.shape) == 1:
                coef_array = np.expand_dims(coef_array, axis=0)

        n = len(coef_array)
        blades = np.empty((n, n_shapes, n_landmarks, dim))
        for k, c in enumerate(coef_array):
            print(f'Perturbing blade {k+1}')
            while True:
                new_blade = get_new_blade(c)
                if not intersection_exist_in_blade(new_blade) or coef is not None:
                    break
                else:
                    print('Generating new coef')
                    c = self.sample_coef()
                    coef_array[k] = c
            blades[k] = new_blade
        if n == 1:
            return blades.squeeze(axis=0), coef_array
        return blades, coef_array


class AffineTransformPerturbation:
    def __init__(self, grid, values, centers, lengthscale, noise_func, noise_params):

        self.mean = PchipInterpolator(grid, values)
        self.centers = centers.reshape(10, -1)
        self.lengthscale = lengthscale

        self.noise_func = NoiseFunc(noise_func, noise_params)

    def rbf(self, x):
        x = np.tile(x, (len(self.centers), 1))
        return np.exp(-(x - self.centers) ** 2 / 2 / self.lengthscale ** 2)

    def generate_perturbed_curves(self, x, n):

        coef = np.random.normal(size=(n, len(self.centers)))
        y = self.noise_func(x) * (coef @ self.rbf(x))
        return y + self.mean(x), coef

    def generate_from_coef(self, x, coef):
        return self.noise_func(x) * (coef @ self.rbf(x)) + self.mean(x)


class NoiseFunc:
    def __init__(self, noise_func, noise_params):
        if noise_func == 'ReLU':
            self.noise_func = self.ReLU
            self.start = noise_params[0]
            self.scale = noise_params[1]
        elif noise_func == 'sigmoid':
            self.noise_func = self.sigmoid
            self.start = noise_params[0]
            self.scale1 = noise_params[1]
            self.scale2 = noise_params[2]
        elif noise_func == 'ReLU_0':
            self.noise_func = self.ReLU_0
            self.start = noise_params[0]
            self.scale = noise_params[1]

        else:
            print("Error: Unknown Noise function!")
            exit()

    def ReLU(self, x):
        return self.scale * (x - self.start) * (x > self.start)

    def ReLU_0(self, x):
        noise = self.scale * (x - self.start) * (x > self.start)
        noise[-1] = 0
        return noise

    def sigmoid(self, x):
        return self.scale1 / (1 + np.exp(-(x - self.start) * self.scale2))

    def __call__(self, x):
        return self.noise_func(x)
