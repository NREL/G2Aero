import os
import numpy as np
from unittest import TestCase
from g2aero.PGA import *
from g2aero.Grassmann import procrustes
from g2aero.SPD import polar_decomposition

PGASPASE_FILENAME = os.path.join(os.getcwd(), "data", 'pga_space', 'CST_Gr_PGA.npz')
SHAPES_FILENAME = os.path.join(os.getcwd(), "data", 'airfoils', 'CST_shapes_TE_gap.npz')

class Test(TestCase):

    def setUp(self):

        self.pga = Grassmann_PGAspace.load_from_file(PGASPASE_FILENAME)
        shapes = np.load(SHAPES_FILENAME)['shapes']
        self.X, _, _ = polar_decomposition(shapes[np.random.randint(len(shapes))])

    def test_PGA2GrassmannShape(self):
        t = self.pga.gr_shapes2PGA(self.X)
        X_new = self.pga.PGA2gr_shape(t)
        X_new = X_new @ procrustes(X_new, self.X)
        np.testing.assert_almost_equal(self.X, X_new, decimal=10, err_msg='', verbose=True)