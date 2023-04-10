import numpy as np
from scipy.interpolate import PchipInterpolator
import g2aero.Grassmann as gr
from g2aero.utils import check_selfintersect
import g2aero.SPD as spd


class Grassmann_PGAspace:
    def __init__(self, Vh, M_mean, b_mean, karcher_mean, t):
        # self.n_modes = Vh.shape[0]
        self.n_landmarks, self.ndim = karcher_mean.shape
        self.Vh = Vh
        self.karcher_mean = karcher_mean
        self.t = t
        self.M_mean = M_mean
        self.b_mean = b_mean

        # for coefficients sampling.
        self.axis_min = np.min(t, axis=0)
        self.axis_max = np.max(t, axis=0)
        r_min = np.abs(np.quantile(t, 0.1, axis=0))
        r_max = np.abs(np.quantile(t, 0.9, axis=0))
        self.radius = np.max(np.array([r_min, r_max]), axis=0)

    @classmethod
    def load_from_file(cls, filename, n_modes=None):
        """Loading PGA space from the .npz file. Can load trancated space 
        (n dimensions defined by `n_modes`)

        :param filename: path to the file
        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :return: PGA_space class instance
        """
        pga_dict = np.load(filename)
        Vh = pga_dict['Vh'][:n_modes, :]
        t = pga_dict['coords'][:, :n_modes]
        pga_space = cls(Vh, pga_dict['M_mean'], pga_dict['b_mean'], pga_dict['karcher_mean'], t)
        return pga_space

    def save_to_file(self, filename):
        np.savez(filename, Vh=self.Vh, karcher_mean=self.karcher_mean, 
                 M_mean=self.M_mean, b_mean=self.b_mean, coords=self.t)

    @classmethod
    def create_from_dataset(cls, phys_shapes, n_modes=None, method='SPD'):
        """Create PGA space by using the dataset of shapes and performing PGA.

        :param phys_shapes: (n_shapes, n_landmarks, ndim) array of dataset of shapes
        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :param method: _description_, defaults to 'SPD'
        :return: _description_
        """
        if method == 'SPD':
            shapes_gr, M, b = spd.polar_decomposition(phys_shapes)
            M_mean = spd.Karcher(M)  #M_mean is default to M_karcher
        elif method == 'LA-transform':
            shapes_gr, M, b = gr.landmark_affine_transform(phys_shapes)
            M_mean == np.mean(M, axis=0)
        
        karcher_mean = gr.Karcher(shapes_gr)
        Vh, S, t = gr.PGA(karcher_mean, shapes_gr, n_coord=n_modes)
        pga_space = cls(Vh, M_mean, np.mean(b, axis=0), karcher_mean, t)
        pga_space.S = S
        return pga_space, t

    def PGA2gr_shape(self, pga_coord, original_shape_gr=None):
        gr_shape = gr.perturb_gr_shape(self.Vh, self.karcher_mean, pga_coord)
        if original_shape_gr is not None:
            R = gr.procrustes(gr_shape, original_shape_gr)
            gr_shape = gr_shape @ R
        return gr_shape

    def PGA2shape(self, pga_coord, M=None, b=None, original_shape_gr=None):
        if M is None:
            M = self.M_mean
        if b is None:
            b = self.b_mean
        gr_shape = gr.perturb_gr_shape(self.Vh, self.karcher_mean, pga_coord)
        # rotate to match with original shape (e.g. for reconstruction error calculation)
        if original_shape_gr is not None:
            R = gr.procrustes(gr_shape, original_shape_gr)
            gr_shape= gr_shape @ R
        return gr_shape @ M + b

    def gr_shapes2PGA(self, shapes_gr):
        return gr.get_PGA_coordinates(shapes_gr, self.karcher_mean, self.Vh.T)
    
    def shapes2PGA(self, shapes, method='SPD', n_modes=None):
        if method == 'SPD':
            shapes_gr, M, b = spd.polar_decomposition(shapes)
        elif method == 'LA-transform':
            shapes_gr, M, b = gr.landmark_affine_transform(shapes)
        t = gr.get_PGA_coordinates(shapes_gr, self.karcher_mean, self.Vh.T)
        return t[:, :n_modes], M, b

    def sample_coef(self, n_modes, n_samples=1):
        k = 0
        coef = np.empty((n_samples, n_modes))
        while k < n_samples:
            c = np.random.uniform(self.axis_min[:n_modes], self.axis_max[:n_modes])
            # Check that c is inside of ellipse (this criteria doesn't work when eigenspaces collapse)
            # if np.sum(c ** 2 / self.radius ** 2) <= 1:
            coef[k] = c
            k += 1
        return coef

    def generate_perturbed_shapes(self, n_modes=None, n=1):
        """Samples PGA coefficients and corresponding shapes.

        Coef are randomly sampled and shapes are checked for self-intersection
        and resample if needed.

        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :param n: number of perturbed shapes, (defualt n=1)
        :return: array of generated perturbed shapes (n, n_landmarks, 2) in physical space
                 array of grassmann shapes (n, n_landmarks, 2) 
                 array of PGA coordinates corresponding to perturbations (n, n_modes)
        """
        if n_modes is None:
            n_modes = self.Vh.shape[0]
        coef_array = self.sample_coef(n_samples=n, n_modes=n_modes)
        gr_samples = np.empty((n, self.n_landmarks, self.ndim))
        phys_samples = np.empty_like(gr_samples)
        for i, c in enumerate(coef_array):
            while True:
                gr_samples[i] = gr.perturb_gr_shape(self.Vh, self.karcher_mean, c)
                if not check_selfintersect(gr_samples[i]):
                    break
                else:
                    c = self.sample_coef(n_modes=n_modes)
                    print(f"WARNING: New shape {i} has intersection! Generating new coef")
                    coef_array[i] = c
            phys_samples[i] = gr_samples[i] @ self.M_mean + self.b_mean
        if n == 1:
            return phys_samples.squeeze(axis=0), gr_samples.squeeze(axis=0),  coef_array
        return phys_samples, gr_samples, coef_array

    def generate_perturbed_blade(self, blade, n_modes=None, coef=None, n=1):
        """ Generates perturbed blades.

        If coef are sampled (not given) checks for intersection
        in generated shape and resample if needed.

        :param blade: array of grassmann shapes for baseline blade
        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :param coef: perturbation coefficients (sampled if not given)
        :param n: number of perturbations (if need to sample)
        :return: array of perturbed blades (in grassmann coordinates) (shape=(n, n_shapes, n_landmarks, 2)) and 
                 array of PGA corresponding to perturbations (n, n_modes)
        """
        n_shapes, n_landmarks, ndim = blade.shape
        if n_landmarks!=self.n_landmarks or ndim!=self.ndim:
            raise ValueError("n_landmark or dimension of the blade shapes don't match PGA space dimensions")
        if n_modes is None:
                n_modes = self.Vh.shape[0]

        def get_new_blade(c):
            new_blade = np.empty((n_shapes, n_landmarks, ndim))
            vector = (c @ self.Vh).reshape(-1, 2)
            for i, shape in enumerate(blade):
                direction = gr.log(self.karcher_mean, shape)
                new_vector = gr.parallel_translate(self.karcher_mean, direction, vector)
                new_blade[i] = gr.exp(1, shape, new_vector)
            return new_blade

        def intersection_exist_in_blade(blade):
            for i, shape in enumerate(blade):
                if check_selfintersect(shape):
                    print(f"WARNING: New shape {i} has intersection!")
                    return True
            return False

        if coef is None:
            coef_array = self.sample_coef(n_samples=n, n_modes=n_modes)
        else:
            coef_array = np.asarray(coef, n_modes)
            if len(coef_array.shape) == 1:
                coef_array = np.expand_dims(coef_array, axis=0)

        n = len(coef_array)
        blades = np.empty((n, n_shapes, n_landmarks, ndim))
        for k, c in enumerate(coef_array):
            print(f'Perturbing blade {k+1}')
            while True:
                new_blade = get_new_blade(c)
                if not intersection_exist_in_blade(new_blade) or coef is not None:
                    break
                else:
                    print('Generating new coef')
                    c = self.sample_coef(n_modes=n_modes)
                    coef_array[k] = c
            blades[k] = new_blade
        if n == 1:
            return blades.squeeze(axis=0), coef_array
        return blades, coef_array


##########################################################################################################################
#
#
#
##########################################################################################################################
class SPD_PGAspace:

    def __init__(self, Vh, shape_mean, b_mean, karcher_mean, t):
        self.n_landmarks, self.ndim = karcher_mean.shape
        self.Vh = Vh
        self.karcher_mean = karcher_mean
        self.t = t
        self.shape_mean = shape_mean
        self.b_mean = b_mean

    @classmethod
    def load_from_file(cls, filename, n_modes=None):
        """Loading PGA space from the .npz file. Can load trancated space 
        (n dimensions defined by `n_modes`)

        :param filename: path to the file
        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :return: PGA_space class instance
        """
        
        pga_dict = np.load(filename)
        Vh = pga_dict['Vh'][:n_modes, :]
        t = pga_dict['coords'][:, :n_modes]
        pga_space = cls(Vh, pga_dict['shape_mean'], pga_dict['b_mean'], pga_dict['karcher_mean'], t)
        return pga_space

    def save_to_file(self, filename):
        np.savez(filename, Vh=self.Vh, karcher_mean=self.karcher_mean, 
                 shape_mean=self.shape_mean, b_mean=self.b_mean, coords=self.t)

    @classmethod
    def create_from_dataset(cls, phys_shapes):
        """Create PGA space by using the dataset of shapes and performing PGA.

        :param phys_shapes: (n_shapes, n_landmarks, ndim) array of dataset of shapes
        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :param method: _description_, defaults to 'SPD'
        :return: _description_
        """
        shapes_gr, P, b = spd.polar_decomposition(phys_shapes)
        karcher_mean = spd.Karcher(P)

        Vh, S, t_spd = spd.PGA(karcher_mean, P)
        shape_karcher = gr.Karcher(shapes_gr)
        spd_pga_space = cls(Vh, shape_karcher, np.mean(b, axis=0), karcher_mean, t_spd)
        spd_pga_space.S = S
        return spd_pga_space, t_spd
    
    def recreate_data(self):
        spd_elements = np.empty((len(self.t), 2, 2))
        for i, ti in enumerate(self.t):
            spd_elements[i] = self.PGA2spd(ti)
        return spd_elements

    def PGA2spd(self, pga_coord):
        return spd.perturb_mu(self.Vh, self.karcher_mean, pga_coord)


##########################################################################################################################
#
#
#
##########################################################################################################################
class SPD_TangentSpace:
    def __init__(self, karcher_mean, t, shape_mean, b_mean, ):
        
        self.n_landmarks, self.ndim = karcher_mean.shape
        self.karcher_mean = karcher_mean
        self.t = t
        self.shape_mean = shape_mean
        self.b_mean = b_mean

    @classmethod
    def load_from_file(cls, filename):
        """Loading SPD tangent space from the .npz file. 

        :param filename: path to the file
        :return: SPD_TangentSpace class instance
        """
        
        pga_dict = np.load(filename)
        pga_space = cls(pga_dict['karcher_mean'], pga_dict['coords'], pga_dict['shape_mean'], pga_dict['b_mean'], )
        return pga_space

    def save_to_file(self, filename):
        np.savez(filename, karcher_mean=self.karcher_mean, 
                 shape_mean=self.shape_mean, b_mean=self.b_mean, coords=self.t)

    @classmethod
    def create_from_dataset(cls, phys_shapes):
        """Create PGA space by using the dataset of shapes and performing PGA.

        :param phys_shapes: (n_shapes, n_landmarks, ndim) array of dataset of shapes
        :param n_modes: dimensions of trancated PGA space, defaults to None (using full dimension)
        :param method: _description_, defaults to 'SPD'
        :return: _description_
        """
        shapes_gr, P, b = spd.polar_decomposition(phys_shapes)
        karcher_mean = spd.Karcher(P)
        t_spd = spd.tangent_space(karcher_mean, P)
        shape_karcher = gr.Karcher(shapes_gr)
        spd_tan_space = cls(karcher_mean, t_spd, shape_karcher, np.mean(b, axis=0))
        return spd_tan_space, t_spd
    
    def recreate_data(self):
        spd_elements = np.empty((len(self.t), 2, 2))
        for i, ti in enumerate(self.t):
            spd_elements[i] = self.tan2spd(ti)
        return spd_elements

    def tan2spd(self, pga_coord):
        direction = spd.vecinv(pga_coord)
        spd_element = spd.exp(1, self.karcher_mean, direction)
        return spd_element








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
