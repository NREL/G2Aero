import os
from unittest import TestCase
from g2aero.Grassmann import *


class Test(TestCase):

    def setUp(self):
        self.X_phys = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'X_phys.txt'))
        self.Y_phys = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'Y_phys.txt'))
        self.X = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'X_gr_la.txt'))
        self.Y = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'Y_gr_la.txt'))
        self.vector = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'vector_gr.txt'))

    def test_distance(self):
        assert distance(self.X, self.Y) >= 0.0
        np.testing.assert_almost_equal(distance(self.X, self.Y), distance(self.Y, self.X), decimal=10, err_msg='', verbose=True)

    def test_log_exp(self):
        # check Exp(Log)
        direction = log(self.X, self.Y)
        new_Y = exp(1, self.X, direction)
        # R = procrustes(new_Y, self.Y)
        # new_Y = new_Y @ R
        np.testing.assert_almost_equal(self.Y, new_Y, decimal=10, err_msg='', verbose=True)
        np.testing.assert_almost_equal(distance(self.Y, new_Y), 0.0, decimal=10, err_msg='', verbose=True)
        #check Log(Exp)
        Y1 = exp(1, self.X, self.vector)
        new_vector = log(self.X, Y1)
        np.testing.assert_almost_equal(self.vector, new_vector, decimal=10, err_msg='', verbose=True)

    def test_parallel_translate(self):
        direction = log(self.X, self.Y)
        inner_product = np.inner(direction, self.vector)
        new_vector = parallel_translate(self.X, direction, self.vector)
        direction_back = log(self.Y, self.X)
        vector_again = parallel_translate(self.Y, direction_back, new_vector)
        new_inner_product = np.inner(direction, vector_again)
        np.testing.assert_almost_equal(inner_product, new_inner_product, decimal=10, err_msg='', verbose=True)
        np.testing.assert_almost_equal(self.vector, vector_again, decimal=10, err_msg='', verbose=True)

    # def test_pga2shape_shape2pga(self):
        # Vh, _, _ = PGA
        # pga = get_PGA_coordinates(X, Y)

    def test_landmark_affine_transform(self):

        # single shape
        shapes_gr, M, b = landmark_affine_transform(self.Y_phys)
        new_Y = shapes_gr @ M + b
        np.testing.assert_almost_equal(self.Y_phys, new_Y, decimal=10, err_msg='', verbose=True)

        # # array of shapes
        shapes = np.vstack((self.X, self.Y))
        shapes_gr, M, b = landmark_affine_transform(shapes)
        new_shapes = shapes_gr @ M + b
        np.testing.assert_almost_equal(shapes, new_shapes, decimal=10, err_msg='', verbose=True)

    def test_procrustes(self):

        # Recover angle
        R1 = np.array([[np.cos(np.pi/2), np.sin(np.pi/2)], [- np.sin(np.pi/2), np.cos(np.pi/2)]])
        new_R1 = procrustes(self.Y, self.Y @ R1)
        np.testing.assert_almost_equal(R1, new_R1, decimal=10, err_msg='', verbose=True)


