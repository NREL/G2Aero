import numpy as np
import g2aero.Grassmann as gr
import g2aero.SPD as spd
from g2aero.PGA import Grassmann_PGAspace, SPD_TangentSpace


class Dataset:
    def __init__(self, phys_shapes, method='SPD'):
        self.n_shapes = len(phys_shapes)
        if method == 'SPD':
            self.shapes_gr, self.M, self.b = spd.polar_decomposition(phys_shapes)
        elif method == 'LA-transform':
            self.shapes_gr, self.M, self.b = gr.landmark_affine_transform(phys_shapes)


class ProductManifold():
    def __init__(self, phys_shapes, n_modes_gr_pga=None, gr_pga_space=None):
        self.n_shapes, self.n_landmarks, self.ndim = phys_shapes.shape
        self.data = Dataset(phys_shapes, method='SPD')
        self.b_mean = np.mean(self.data.b, axis=0)
        
        if gr_pga_space is None:
            self.gr_karcher_mean = gr.Karcher(self.data.shapes_gr)
        else:
            self.gr_karcher_mean = gr_pga_space.karcher_mean
        
        self.M_karcher_mean = spd.Karcher(self.data.M)

        if gr_pga_space is None:
            print("\nGrassmann manifold PGA")
            Vh, S, t = gr.PGA(self.gr_karcher_mean, self.data.shapes_gr, n_coord=n_modes_gr_pga)
            self.data.t_gr = t
            self.gr_pga_space = Grassmann_PGAspace(Vh, self.M_karcher_mean, self.b_mean, self.gr_karcher_mean, t)
            self.gr_pga_space.S = S
        else:
            self.gr_pga_space = gr_pga_space

        print("SPD manifold Tangent space")
        self.data.t_spd = spd.tangent_space(self.M_karcher_mean, self.data.M)
        self.spd_tan_space = SPD_TangentSpace(self.M_karcher_mean, self.data.t_spd, self.gr_karcher_mean, self.b_mean)

