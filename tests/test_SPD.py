import os
from unittest import TestCase
from g2aero.SPD import *


class Test(TestCase):

    def setUp(self):
        self.X_phys = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'X_phys.txt'))
        self.Y_phys = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'Y_phys.txt'))

        self.P = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'P.txt'))
        self.D = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'D.txt'))
        self.S = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'S.txt'))
        self.vector = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'vector_spd.txt'))

    def test_log_exp(self):
        # check Exp(Log)
        direction = log(self.S, self.D)
        new_D = exp(1, self.S, direction)
        np.testing.assert_almost_equal(self.D, new_D, decimal=10, err_msg='', verbose=True)
        #check Log(Exp)
        D1 = exp(1, self.P, self.vector)
        new_vector = log(self.P, D1)
        np.testing.assert_almost_equal(self.vector, new_vector, decimal=10, err_msg='', verbose=True)

    def test_polar_decomposition(self):

        # single shape
        shapes_gr, P, b = polar_decomposition(self.X_phys)
        new_X = shapes_gr @ P + b
        np.testing.assert_almost_equal(self.X_phys, new_X, decimal=10, err_msg='', verbose=True)

        # array of shapes
        shapes = np.vstack((self.X_phys, self.Y_phys))
        shapes_gr, P, b = polar_decomposition(shapes)
        new_shapes = shapes_gr @ P + b
        np.testing.assert_almost_equal(shapes, new_shapes, decimal=10, err_msg='', verbose=True)

        #TODO: check orthonormality of columns
