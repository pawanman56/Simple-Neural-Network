"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the implementation of the networks' activation functions.
Students may use the functions below to validate their solutions of the proposed tasks.

@author: Sebastian Doerrich
@copyright: Copyright (c) 2022, Chair of Explainable Machine Learning (xAI), Otto-Friedrich University of Bamberg
@credits: [Christian Ledig, Sebastian Doerrich]
@license: CC BY-SA
@version: 1.0
@python: Python 3
@maintainer: Sebastian Doerrich
@email: sebastian.doerrich@uni-bamberg.de
@status: Production
"""

import unittest
import numpy as np
from models.softmax_regression import SoftmaxRegression


class TestActivation(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """ Initialize the tests. """
        self.model = SoftmaxRegression()

    def test_sigmoid(self):
        """ Test the implementation of the sigmoid function. """

        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.1841628, 0.4218198],
                      [0.42978908, 0.26740977],
                      [0.66023782, 0.77794766],
                      [0.16133995, 0.71140804]])
        outs = self.model.sigmoid(x)
        diff = np.sum(np.abs((outs - y)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_sigmoid_dev(self):
        """ Test the implementation of the derivative of the sigmoid function. """

        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.15024686, 0.24388786],
                      [0.24507043, 0.19590178],
                      [0.22432384, 0.1727451 ],
                      [0.13530937, 0.20530664]])

        outs = self.model.sigmoid_dev(x)
        diff = np.sum(np.abs((outs - y)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_relu(self):
        """ Test the implementation of the ReLU function. """

        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.66435418, 1.2537461],
                      [0.0, 0.90223236]])
        out = self.model.ReLU(x)
        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_relu_dev(self):
        """ Test the implementation of the derivative of the ReLU function. """

        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [1., 1.],
                      [0.0, 1.]])
        out = self.model.ReLU_dev(x)
        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)

    def test_softmax(self):
        """ Test the implementation of the softmax function. """

        x = np.array([[-1.48839468, -0.31530738],
                      [-0.28271176, -1.00780433],
                      [0.66435418, 1.2537461],
                      [-1.64829182, 0.90223236]])
        y = np.array([[0.23629739, 0.76370261],
                      [0.67372745, 0.32627255],
                      [0.35677439, 0.64322561],
                      [0.07239128, 0.92760872]])

        out = self.model.softmax(x)

        diff = np.sum(np.abs((y - out)))
        self.assertAlmostEqual(diff, 0, places=7)
