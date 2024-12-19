#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'optimization.ipynb'-jupyter notebook.
That notebook lets you analyze different optimization methods on some sample data.
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

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# Import packages
import numpy as np
import unittest
import argparse

# Import own files
import nbimporter  # Necessary to be able to use equations of ipynb-notebooks in python-files
import optimization


class TestGradientDescent(unittest.TestCase):
    """
    The class contains all test cases for task "1 - Gradient Descent".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        learning_rate = 0.01

        W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
        W2, b2 = np.random.randn(3, 3), np.random.randn(3, 1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        dW1, db1 = np.random.randn(2, 3), np.random.randn(2, 1)
        dW2, db2 = np.random.randn(3, 3), np.random.randn(3, 1)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        # Create the student version
        self.stud_vers = optimization.gradient_descent(parameters, grads, learning_rate)

        # Load the references
        self.exp_vers = {'W1': np.array([[ 1.63535156, -0.62320365, -0.53718766],
                                         [-1.07799357,  0.85639907, -2.29470142]]),
                         'b1': np.array([[ 1.74604067],
                                         [-0.75184921]]),
                         'W2': np.array([[ 0.32171798, -0.25467393,  1.46902454],
                                         [-2.05617317, -0.31554548, -0.3756023 ],
                                         [ 1.1404819 , -1.09976462, -0.1612551 ]]),
                         'b2': np.array([[-0.88020257],
                                         [ 0.02561572],
                                         [ 0.57539477]])}

    def test_parameters(self):
        """ Test the parameter dictionary. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'parameters' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'parameters' is not correct!")

    def test_W1(self):
        """ Test the updated weights of layer 1. """

        self.assertIn('W1', self.stud_vers.keys(), "Weights for layer 1 are missing!")

        stud_vers = self.stud_vers['W1']
        exp_vers = self.exp_vers['W1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W1' is not correct!")

    def test_b1(self):
        """ Test the updated bias of layer 1. """

        self.assertIn('b1', self.stud_vers.keys(), "Bias for layer 1 is missing!")

        stud_vers = self.stud_vers['b1']
        exp_vers = self.exp_vers['b1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b1' is not correct!")

    def test_W2(self):
        """ Test the updated weights of layer 2. """

        self.assertIn('W2', self.stud_vers.keys(), "Weights for layer 2 are missing!")

        stud_vers = self.stud_vers['W2']
        exp_vers = self.exp_vers['W2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W2' is not correct!")

    def test_b2(self):
        """ Test the updated bias of layer 2. """

        self.assertIn('b2', self.stud_vers.keys(), "Bias for layer 2 is missing!")

        stud_vers = self.stud_vers['b2']
        exp_vers = self.exp_vers['b2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b2' is not correct!")


class TestShuffling(unittest.TestCase):
    """
    The class contains all test cases for the shuffling task of task "1.4 - Mini-Batch Gradient Descent".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Test 1
        # Create the parameters
        seed = 1
        X_1 = np.array([[1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5]])
        Y_1 = np.array([[1, 0, 1, 0, 0]])

        # Create the student version
        stud_1_o1, stud_1_o2 = optimization.shuffle(X_1, Y_1, seed)
        self.stud_vers_1 = {"shuffled_X": stud_1_o1, "shuffled_Y": stud_1_o2}

        # Load the references
        self.exp_vers_1 = \
            {"shuffled_X": np.array([[3, 2, 5, 1, 4],
                                     [3, 2, 5, 1, 4],
                                     [3, 2, 5, 1, 4],
                                     [3, 2, 5, 1, 4]]),
             "shuffled_Y": np.array([[1, 0, 0, 1, 0]])}

        # Test 2
        # Create the parameters
        seed = 1
        X_2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                        [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
                        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]).T
        Y_2 = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]])

        # Create the student version
        np.random.seed(seed)
        stud_2_o1, stud_2_o2 = optimization.shuffle(X_2, Y_2, seed)
        self.stud_vers_2 = {"shuffled_X": stud_2_o1, "shuffled_Y": stud_2_o2}

        # Load the references
        self.exp_vers_2 = \
            {"shuffled_X": np.array([[8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5],
                                     [8, 2, 6, 7, 1, 0, 4, 3, 5]]),
             "shuffled_Y": np.array([[0, 1, 0, 0, 1, 1, 1, 1, 0]])}

    def test_shuffled_X_1(self):
        """ Test shuffling the input data (X) for test case 1. """

        stud_vers = self.stud_vers_1["shuffled_X"]
        exp_vers = self.exp_vers_1["shuffled_X"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'shuffled_X' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'shuffled_X' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'shuffled_X' is not correct!")

    def test_shuffled_Y_1(self):
        """ Test shuffling the labels (Y) for test case 1. """

        stud_vers = self.stud_vers_1["shuffled_Y"]
        exp_vers = self.exp_vers_1["shuffled_Y"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'shuffled_Y' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'shuffled_Y' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'shuffled_Y' is not correct!")

    def test_shuffled_X_2(self):
        """ Test shuffling the input data (X) for test case 2. """

        stud_vers = self.stud_vers_2["shuffled_X"]
        exp_vers = self.exp_vers_2["shuffled_X"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'shuffled_X' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'shuffled_X' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'shuffled_X' is not correct!")

    def test_shuffled_Y_2(self):
        """ Test shuffling the labels (Y) for test case 2. """

        stud_vers = self.stud_vers_2["shuffled_Y"]
        exp_vers = self.exp_vers_2["shuffled_Y"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'shuffled_Y' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'shuffled_Y' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'shuffled_Y' is not correct!")


class TestMiniBatches(unittest.TestCase):
    """
    The class contains all test cases for the mini-batch task of task "1.4 - Mini-Batch Gradient Descent".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Test 1
        # Create the parameters
        seed_1 = 1
        np.random.seed(1)
        mini_batch_size_1 = 2
        X_1 = np.random.randn(7, 3)
        np.random.seed(1)
        Y_1 = np.random.randn(1, 3) < 0.5

        # Create the student version
        self.stud_vers_1 = optimization.random_mini_batches(X_1, Y_1, mini_batch_size_1, seed_1)

        # Load the references
        self.exp_vers_1 = [(np.array([[ 1.62434536, -0.52817175],
                                   [-1.07296862, -2.3015387 ],
                                   [ 1.74481176,  0.3190391 ],
                                   [-0.24937038, -2.06014071],
                                   [-0.3224172 ,  1.13376944],
                                   [-1.09989127, -0.87785842],
                                   [ 0.04221375, -1.10061918]]),
                            np.array([[False,  True]])),
                           (np.array([[-0.61175641],
                                      [ 0.86540763],
                                      [-0.7612069 ],
                                      [ 1.46210794],
                                      [-0.38405435],
                                      [-0.17242821],
                                      [ 0.58281521]]),
                            np.array([[True]]))]

        # Test 2
        # Create the parameters
        seed_2 = 1
        self.mini_batch_size_2 = 64
        self.nx = 12288
        self.m = 148
        X_2 = np.array([x for x in range(self.nx * self.m)]).reshape((self.m, self.nx)).T
        np.random.seed(1)
        Y_2 = np.random.randn(1, self.m) < 0.5

        # Create the student version
        self.stud_vers_2 = optimization.random_mini_batches(X_2, Y_2, self.mini_batch_size_2, seed_2)

        # Load the references
        self.exp_vers_2 = \
            {"nr_mini_batches": 3,
             "X_shape": (12288, 64),
             "Y_shape": (1, 64),
             }

    def test_mini_batch_1(self):
        """ Test the mini-batch for test case 1. """

        stud_vers = self.stud_vers_1
        exp_vers = self.exp_vers_1

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'mini_batches' is not correct!")
        self.assertEqual(len(stud_vers), len(exp_vers), "Length of 'mini_batches' is not correct!")

    def test_mini_batch_entries_type_1(self):
        """ Test the data type of the mini-batch elements for test case 1. """

        for mb_stud, mb_exp in zip(self.stud_vers_1, self.exp_vers_1):
            stud_vers_X, exp_vers_X = mb_stud[0], mb_exp[0]
            stud_vers_Y, exp_vers_Y = mb_stud[1], mb_exp[1]

            self.assertEqual(type(stud_vers_X), type(stud_vers_X), "Type of data elements is not correct!")
            self.assertEqual(type(stud_vers_Y), type(stud_vers_Y), "type of label elements is not correct!")

    def test_mini_batch_entries_shape_1(self):
        """ Test the shape of the mini-batch elements for test case 1. """

        for mb_stud, mb_exp in zip(self.stud_vers_1, self.exp_vers_1):
            stud_vers_X, exp_vers_X = mb_stud[0], mb_exp[0]
            stud_vers_Y, exp_vers_Y = mb_stud[1], mb_exp[1]

            self.assertEqual(stud_vers_X.shape, exp_vers_X.shape, "Shape of data elements is not correct!")
            self.assertEqual(stud_vers_Y.shape, exp_vers_Y.shape, "Shape of label elements is not correct!")

    def test_mini_batch_entries_1(self):
        """ Test the mini-batch elements for test case 1. """

        for mb_stud, mb_exp in zip(self.stud_vers_1, self.exp_vers_1):
            stud_vers_X, exp_vers_X = mb_stud[0], mb_exp[0]
            stud_vers_Y, exp_vers_Y = mb_stud[1], mb_exp[1]

            self.assertTrue(np.allclose(stud_vers_X, exp_vers_X, atol=0.0001), "Data elements are not correct!")
            self.assertTrue(np.allclose(stud_vers_Y, exp_vers_Y, atol=0.0001), "Label elements are not correct!")

    def test_mini_batch_2(self):
        """ Test the mini-batch for test case 2. """

        stud_vers = self.stud_vers_2
        exp_vers = self.exp_vers_2["nr_mini_batches"]

        self.assertIsInstance(stud_vers, list, "Type of 'mini_batches' is not correct!")
        self.assertEqual(len(stud_vers), exp_vers, "Length of 'mini_batches' is not correct!")

    def test_mini_batch_entries_type_2(self):
        """ Test the data type of the mini-batch elements for test case 2. """

        for mb_stud in self.stud_vers_2:
            stud_vers_X, stud_vers_Y = mb_stud[0], mb_stud[1]

            self.assertIsInstance(stud_vers_X, np.ndarray, "Type of data elements is not correct!")
            self.assertIsInstance(stud_vers_Y, np.ndarray, "type of label elements is not correct!")

    def test_mini_batch_entries_shape_2(self):
        """ Test the shape of the mini-batch elements for test case 2. """

        exp_vers_X = self.exp_vers_2["X_shape"]
        exp_vers_Y = self.exp_vers_2["Y_shape"]

        # Test for all full mini-batches
        for mb_stud in self.stud_vers_2[:-1]:
            stud_vers_X, stud_vers_Y = mb_stud[0], mb_stud[1]

            self.assertEqual(stud_vers_X.shape, exp_vers_X, "Shape of data elements is not correct!")
            self.assertEqual(stud_vers_Y.shape, exp_vers_Y, "Shape of label elements is not correct!")

        # Test for last mini-batch
        stud_vers_X, stud_vers_Y = self.stud_vers_2[-1][0], self.stud_vers_2[-1][1]
        self.assertEqual(stud_vers_X.shape, (self.nx, self.m % self.mini_batch_size_2), "Shape of last mini-batch's data element is not correct!")
        self.assertEqual(stud_vers_Y.shape, (1, self.m % self.mini_batch_size_2), "Shape of last mini-batch's data element is not correct!")

    def test_mini_batch_entries_2(self):
        """ Test the mini-batch elements for test case 2. """

        stud_vers_X1, stud_vers_X2 = self.stud_vers_2[0][0][0][0:3], self.stud_vers_2[-1][0][-1][0:3]
        exp_vers_X1, exp_vers_X2 = [1228800, 1486848, 663552], [282623, 761855, 786431]

        self.assertTrue(np.allclose(stud_vers_X1, exp_vers_X1, atol=0.0001), "Data elements are not correct!")
        self.assertTrue(np.allclose(stud_vers_X2, exp_vers_X2, atol=0.0001), "Label elements are not correct!")


class TestInitializingVelocity(unittest.TestCase):
    """
    The class contains all test cases for task "2.1 - Initialize the Velocity v".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        # Create the student version
        self.stud_vers = optimization.initialize_velocity(parameters)

        # Load the references
        self.exp_vers = {'dW1': np.array([[0., 0., 0.],
                                          [0., 0., 0.]]),
                         'db1': np.array([[0.],
                                          [0.]]),
                         'dW2': np.array([[0., 0., 0.],
                                          [0., 0., 0.],
                                          [0., 0., 0.]]),
                         'db2': np.array([[0.],
                                          [0.],
                                          [0.]])}

    def test_parameters(self):
        """ Test the velocity dictionary. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'v' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'v' is not correct!")

    def test_dW1(self):
        """ Test the velocity of dW1. """

        self.assertIn('dW1', self.stud_vers.keys(), "Velocity of dW1 is missing!")

        stud_vers = self.stud_vers['dW1']
        exp_vers = self.exp_vers['dW1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW1' is not correct!")

    def test_db1(self):
        """ Test the velocity of db1. """

        self.assertIn('db1', self.stud_vers.keys(), "Velocity of db1 is missing!")

        stud_vers = self.stud_vers['db1']
        exp_vers = self.exp_vers['db1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db1' is not correct!")

    def test_dW2(self):
        """ Test the velocity of dW2. """

        self.assertIn('dW2', self.stud_vers.keys(), "Velocity of dW2 is missing!")

        stud_vers = self.stud_vers['dW2']
        exp_vers = self.exp_vers['dW2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW2' is not correct!")

    def test_db2(self):
        """ Test the velocity of db2. """

        self.assertIn('db2', self.stud_vers.keys(), "Velocity of db2 is missing!")

        stud_vers = self.stud_vers['db2']
        exp_vers = self.exp_vers['db2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db2' is not correct!")


class TestMomentum(unittest.TestCase):
    """
    The class contains all test cases for task "2.2 - Update the Model’s Parameters with Momentum".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        beta = 0.9
        learning_rate = 0.01

        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)

        dW1 = np.random.randn(2, 3)
        db1 = np.random.randn(2, 1)
        dW2 = np.random.randn(3, 3)
        db2 = np.random.randn(3, 1)

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        v = {'dW1': np.array([[0.,  0.,  0.],
                              [0.,  0.,  0.]]),
             'dW2': np.array([[0.,  0.,  0.],
                              [0.,  0.,  0.],
                              [0.,  0.,  0.]]),
             'db1': np.array([[0.],
                              [0.]]),
             'db2': np.array([[0.],
                              [0.],
                              [0.]])}

        # Create the student version
        stud_o1, stud_o2 = optimization.update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
        self.stud_vers = {"updated_params": stud_o1,
                          "updated_v": stud_o2}

        # Load the references
        self.exp_vers = \
            {
                "updated_params":
                    {
                        'W1': np.array([[ 1.62544598, -0.61290114, -0.52907334],
                                        [-1.07347112,  0.86450677, -2.30085497]]),
                        'b1': np.array([[ 1.74493465],
                                        [-0.76027113]]),
                        'W2': np.array([[ 0.31930698, -0.24990073,  1.4627996 ],
                                        [-2.05974396, -0.32173003, -0.38320915],
                                        [ 1.13444069, -1.0998786 , -0.1713109 ]]),
                        'b2': np.array([[-0.87809283],
                                        [ 0.04055394],
                                        [ 0.58207317]])
                    },
                "updated_v":
                    {
                        'dW1': np.array([[-0.11006192,  0.11447237,  0.09015907],
                                         [ 0.05024943,  0.09008559, -0.06837279]]),
                        'dW2': np.array([[-0.02678881,  0.05303555, -0.06916608],
                                         [-0.03967535, -0.06871727, -0.08452056],
                                         [-0.06712461, -0.00126646, -0.11173103]]),
                        'db1': np.array([[-0.01228902],
                                         [-0.09357694]]),
                        'db2': np.array([[0.02344157],
                                         [0.16598022],
                                         [0.07420442]])
                    }
            }

    def test_parameters(self):
        """ Test the updated parameter dictionary. """

        stud_vers = self.stud_vers["updated_params"]
        exp_vers = self.exp_vers["updated_params"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'updated_params' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'updated_params' is not correct!")

    def test_W1(self):
        """ Test the updated weights of layer 1. """

        stud_vers = self.stud_vers["updated_params"]
        exp_vers = self.exp_vers["updated_params"]

        self.assertIn('W1', stud_vers.keys(), "Updated weights for layer 1 are missing!")

        stud_vers = stud_vers['W1']
        exp_vers = exp_vers['W1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W1' is not correct!")

    def test_b1(self):
        """ Test the updated bias of layer 1. """

        stud_vers = self.stud_vers["updated_params"]
        exp_vers = self.exp_vers["updated_params"]

        self.assertIn('b1', stud_vers.keys(), "Updated bias for layer 1 is missing!")

        stud_vers = stud_vers['b1']
        exp_vers = exp_vers['b1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b1' is not correct!")

    def test_W2(self):
        """ Test the updated weights of layer 2. """

        stud_vers = self.stud_vers["updated_params"]
        exp_vers = self.exp_vers["updated_params"]

        self.assertIn('W2', stud_vers.keys(), "Updated weights for layer 2 are missing!")

        stud_vers = stud_vers['W2']
        exp_vers = exp_vers['W2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W2' is not correct!")

    def test_b2(self):
        """ Test the updated bias of layer 2. """

        stud_vers = self.stud_vers["updated_params"]
        exp_vers = self.exp_vers["updated_params"]

        self.assertIn('b2', stud_vers.keys(), "Updated bias for layer 2 is missing!")

        stud_vers = stud_vers['b2']
        exp_vers = exp_vers['b2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b2' is not correct!")

    def test_v(self):
        """ Test the updated velocity dictionary. """

        stud_vers = self.stud_vers["updated_v"]
        exp_vers = self.exp_vers["updated_v"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'updated_v' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'updated_v' is not correct!")

    def test_dW1(self):
        """ Test the updated velocity of dW1. """

        stud_vers = self.stud_vers["updated_v"]
        exp_vers = self.exp_vers["updated_v"]

        self.assertIn('dW1', stud_vers.keys(), "Updated velocity of dW1 is missing!")

        stud_vers = stud_vers['dW1']
        exp_vers = exp_vers['dW1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW1' is not correct!")

    def test_db1(self):
        """ Test the updated velocity of db1. """

        stud_vers = self.stud_vers["updated_v"]
        exp_vers = self.exp_vers["updated_v"]

        self.assertIn('db1', stud_vers.keys(), "Updated velocity of db1 is missing!")

        stud_vers = stud_vers['db1']
        exp_vers = exp_vers['db1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db1' is not correct!")

    def test_dW2(self):
        """ Test the updated velocity of dW2. """

        stud_vers = self.stud_vers["updated_v"]
        exp_vers = self.exp_vers["updated_v"]

        self.assertIn('dW2', stud_vers.keys(), "Updated velocity of dW2 is missing!")

        stud_vers = stud_vers['dW2']
        exp_vers = exp_vers['dW2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW2' is not correct!")

    def test_db2(self):
        """ Test the updated velocity of db2. """

        stud_vers = self.stud_vers["updated_v"]
        exp_vers = self.exp_vers["updated_v"]

        self.assertIn('db2', stud_vers.keys(), "Updated velocity of db2 is missing!")

        stud_vers = stud_vers['db2']
        exp_vers = exp_vers['db2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db2' is not correct!")


class TestInitializingAdam(unittest.TestCase):
    """
    The class contains all test cases for task "3.1 - Initialize Adam".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        # Create the student version
        stud_o1, stud_o2 = optimization.initialize_adam(parameters)

        self.stud_vers = {"v": stud_o1,
                          "s": stud_o2}

        # Load the references
        self.exp_vers = \
            {
                "v":
                    {
                        'dW1': np.array([[0., 0., 0.],
                                         [0., 0., 0.]]),
                        'db1': np.array([[0.],
                                         [0.]]),
                        'dW2': np.array([[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.]]),
                        'db2': np.array([[0.],
                                         [0.],
                                         [0.]])},
                "s":
                    {
                        'dW1': np.array([[0., 0., 0.],
                                         [0., 0., 0.]]),
                        'db1': np.array([[0.],
                                         [0.]]),
                        'dW2': np.array([[0., 0., 0.],
                                         [0., 0., 0.],
                                         [0., 0., 0.]]),
                        'db2': np.array([[0.],
                                         [0.],
                                         [0.]])
                    }
            }

    def test_v(self):
        """ Test the initialized dictionary 'v'. """

        stud_vers = self.stud_vers["v"]
        exp_vers = self.exp_vers["v"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'v' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'v' is not correct!")

    def test_v_dW1(self):
        """ Test for v's dW of layer 1. """

        stud_vers = self.stud_vers["v"]
        exp_vers = self.exp_vers["v"]

        self.assertIn('dW1', stud_vers.keys(), "Key 'dW1' is missing in 'v'!")

        stud_vers = stud_vers['dW1']
        exp_vers = exp_vers['dW1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW1' of 'v' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW1' of 'v' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW1' of 'v' is not correct!")

    def test_v_db1(self):
        """ Test for v's db of layer 1. """

        stud_vers = self.stud_vers["v"]
        exp_vers = self.exp_vers["v"]

        self.assertIn('db1', stud_vers.keys(), "Key 'db1' is missing in 'v'!")

        stud_vers = stud_vers['db1']
        exp_vers = exp_vers['db1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db1' of 'v' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db1' of 'v' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db1' of 'v' is not correct!")

    def test_v_dW2(self):
        """ Test for v's dW of layer 2. """

        stud_vers = self.stud_vers["v"]
        exp_vers = self.exp_vers["v"]

        self.assertIn('dW2', stud_vers.keys(), "Key 'dW2' is missing in 'v'!")

        stud_vers = stud_vers['dW2']
        exp_vers = exp_vers['dW2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW2' of 'v' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW2' of 'v' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW2' of 'v' is not correct!")

    def test_v_db2(self):
        """ Test for v's db of layer 2. """

        stud_vers = self.stud_vers["v"]
        exp_vers = self.exp_vers["v"]

        self.assertIn('db2', stud_vers.keys(), "Key 'db2' is missing in 'v'!")

        stud_vers = stud_vers['db2']
        exp_vers = exp_vers['db2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db2' of 'v' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db2' of 'v' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db2' of 'v' is not correct!")

    def test_s(self):
        """ Test the initialized dictionary 's'. """

        stud_vers = self.stud_vers["s"]
        exp_vers = self.exp_vers["s"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 's' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 's' is not correct!")

    def test_s_dW1(self):
        """ Test for s's dW of layer 1. """

        stud_vers = self.stud_vers["s"]
        exp_vers = self.exp_vers["s"]

        self.assertIn('dW1', stud_vers.keys(), "Key 'dW1' is missing in 's'!")

        stud_vers = stud_vers['dW1']
        exp_vers = exp_vers['dW1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW1' of 's' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW1' of 's' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW1' of 's' is not correct!")

    def test_s_db1(self):
        """ Test for s's db of layer 1. """

        stud_vers = self.stud_vers["s"]
        exp_vers = self.exp_vers["s"]

        self.assertIn('db1', stud_vers.keys(), "Key 'db1' is missing in 's'!")

        stud_vers = stud_vers['db1']
        exp_vers = exp_vers['db1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db1' of 's' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db1' of 's' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db1' of 's' is not correct!")

    def test_s_dW2(self):
        """ Test for s's dW of layer 2. """

        stud_vers = self.stud_vers["s"]
        exp_vers = self.exp_vers["s"]

        self.assertIn('dW2', stud_vers.keys(), "Key 'dW2' is missing in 's'!")

        stud_vers = stud_vers['dW2']
        exp_vers = exp_vers['dW2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'dW2' of 's' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'dW2' of 's' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'dW2' of 's' is not correct!")

    def test_s_db2(self):
        """ Test for s's db of layer 2. """

        stud_vers = self.stud_vers["s"]
        exp_vers = self.exp_vers["s"]

        self.assertIn('db2', stud_vers.keys(), "Key 'db2' is missing in 's'!")

        stud_vers = stud_vers['db2']
        exp_vers = exp_vers['db2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'db2' of 's' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'db2' of 's' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'db2' of 's' is not correct!")


class TestAdam(unittest.TestCase):
    """
    The class contains all test cases for task "3.2 - Update Parameters using Adam".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        t = 2
        learning_rate = 0.02
        beta1 = 0.8
        beta2 = 0.888
        epsilon = 1e-2

        self.vi = {'dW1': np.array([[ 0.,  0.,  0.],
                                    [ 0.,  0.,  0.]]),
                   'dW2': np.array([[ 0.,  0.,  0.],
                                    [ 0.,  0.,  0.],
                                    [ 0.,  0.,  0.]]),
                   'db1': np.array([[ 0.],
                                    [ 0.]]),
                   'db2': np.array([[ 0.],
                                    [ 0.],
                                    [ 0.]])}

        self.si = {'dW1': np.array([[ 0.,  0.,  0.],
                                    [ 0.,  0.,  0.]]),
                   'dW2': np.array([[ 0.,  0.,  0.],
                                    [ 0.,  0.,  0.],
                                    [ 0.,  0.,  0.]]),
                   'db1': np.array([[ 0.],
                                    [ 0.]]),
                   'db2': np.array([[ 0.],
                                    [ 0.],
                                    [ 0.]])}
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)

        dW1 = np.random.randn(2, 3)
        db1 = np.random.randn(2, 1)
        dW2 = np.random.randn(3, 3)
        db2 = np.random.randn(3, 1)

        self.parametersi = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        # Create the student version
        stud_o1, stud_o2, stud_o3, stud_o4, stud_o5 =\
            optimization.update_parameters_with_adam(self.parametersi, grads, self.vi, self.si, t, learning_rate, beta1,
                                                     beta2, epsilon)

        self.stud_vers = {"params": stud_o1,
                          "v": stud_o2,
                          "s": stud_o3,
                          "v_corrected": stud_o4,
                          "s_corrected": stud_o5}

        # Load the references
        self.c1 = 1.0 / (1 - beta1**t)
        self.c2 = 1.0 / (1 - beta2**t)

        self.exp_vers = \
            {
                "params":
                    {'W1': np.array([ 1.63942428, -0.6268425,  -0.54320974]),
                     'W2': np.array([ 0.33356139, -0.26425199, 1.47707772]),
                     'b1': np.array([1.75854357]),
                     'b2': np.array([-0.89228024])},
                "v":
                    {'dW1': np.array([-0.22012384,  0.22894474,  0.18031814]),
                     'dW2': np.array([-0.05357762,  0.10607109, -0.13833215]),
                     'db1': np.array([-0.02457805]),
                     'db2': np.array([0.04688314])},
                "s":
                    {'dW1': np.array([0.13567261, 0.14676395, 0.09104097]),
                     'dW2': np.array([8.03757060e-03, 3.15030152e-02, 5.35801947e-02]),
                     'db1': np.array([0.00169142]),
                     'db2': np.array([0.00615448])}
            }

    def test_v(self):
        """ Test the updated dictionary 'v' containing the moving average of the gradients. """

        stud_vers = self.stud_vers["v"]
        exp_vers = self.exp_vers["v"]

        self.assertIsInstance(stud_vers, dict, "Type of 'v' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'v' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in v!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of v['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, self.vi[key].shape, f"Shape of v['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key][0], exp_vers[key], atol=0.0001), f"v['{key}'] of is not correct!")

    def test_v_corrected(self):
        """ Test the bias-corrected first moment estimate 'v_corrected'. """

        stud_vers = self.stud_vers["v_corrected"]
        exp_vers = self.exp_vers["v"]

        self.assertIsInstance(stud_vers, dict, "Type of 'v_corrected' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'v_corrected' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in v_corrected!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of v_corrected['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, self.vi[key].shape, f"Shape of v_corrected['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key][0], exp_vers[key] * self.c1, atol=0.0001), f"v_corrected['{key}'] of is not correct!")

    def test_s(self):
        """ Test the updated dictionary 's' containing the moving average of the squared gradients. """

        stud_vers = self.stud_vers["s"]
        exp_vers = self.exp_vers["s"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 's' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 's' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in s!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of s['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, self.si[key].shape, f"Shape of s['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key][0], exp_vers[key], atol=0.0001), f"s['{key}'] of is not correct!")

    def test_s_corrected(self):
        """ Test the bias-corrected second raw moment estimate 's_corrected'. """

        stud_vers = self.stud_vers["s_corrected"]
        exp_vers = self.exp_vers["s"]

        self.assertIsInstance(stud_vers, dict, "Type of 's_corrected' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 's_corrected' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in s_corrected!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of s_corrected['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, self.vi[key].shape, f"Shape of s_corrected['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key][0], exp_vers[key] * self.c2, atol=0.0001), f"s_corrected['{key}'] of is not correct!")

    def test_params(self):
        """ Test the updated model parameters. """

        stud_vers = self.stud_vers["params"]
        exp_vers = self.exp_vers["params"]

        self.assertIsInstance(stud_vers, dict, "Type of 'params' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'params' is not correct!")

        for key in exp_vers.keys():
            self.assertIn(key, stud_vers.keys(), f"Key '{key}' is missing in params!")
            self.assertIsInstance(stud_vers[key], np.ndarray, f"Type of params['{key}'] is not correct!")
            self.assertEqual(stud_vers[key].shape, self.parametersi[key].shape, f"Shape of params['{key}'] is not correct!")
            self.assertTrue(np.allclose(stud_vers[key][0], exp_vers[key], atol=0.0001), f"params['{key}'] of is not correct!")


class TestExponentialWeightDecay(unittest.TestCase):
    """
    The class contains all test cases for task "6.1 - Decay on Every Iteration".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        learning_rate = 0.5
        epoch_num = 2
        decay_rate = 1

        # Create the student version
        self.stud_vers = optimization.update_lr(learning_rate, epoch_num, decay_rate)

        # Load the reference
        self.exp_vers = 0.16666666666666666

    def test_updated_learning_rate(self):
        """ Test the updated learning rate 'updated_lr'. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertAlmostEqual(stud_vers, exp_vers, delta=0.0001, msg="'updated_lr' is not correct!")


class TestExponentialWeightDecayScheduled(unittest.TestCase):
    """
    The class contains all test cases for task "6.2 - Fixed Interval Scheduling".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Test 1
        # Create the parameters
        learning_rate_1 = 0.5
        epoch_num_1 = 100
        epoch_num_2 = 10
        decay_rate_1 = 1
        time_interval_1 = 100

        # Create the student version
        self.stud_vers_1 = optimization.schedule_lr_decay(learning_rate_1, epoch_num_1, decay_rate_1, time_interval_1)

        # Load the reference
        self.exp_vers_1 = 0.25

        # Test 2
        # Create the parameters
        learning_rate_2 = 0.5
        epoch_num_2 = 10
        decay_rate_2 = 1
        time_interval_2 = 100

        # Create the student version
        self.stud_vers_2 = optimization.schedule_lr_decay(learning_rate_2, epoch_num_2, decay_rate_2, time_interval_2)

        # Load the reference
        self.exp_vers_2 = 0.5

        # Test 3
        # Create the parameters
        learning_rate_3 = 0.3
        epoch_num_3 = 1000
        decay_rate_3 = 0.25
        time_interval_3 = 100

        # Create the student version
        self.stud_vers_3 = optimization.schedule_lr_decay(learning_rate_3, epoch_num_3, decay_rate_3, time_interval_3)

        # Load the reference
        self.exp_vers_3 = 0.085714285

        # Test 4
        # Create the parameters
        learning_rate_4 = 0.3
        epoch_num_4 = 100
        decay_rate_4 = 0.25
        time_interval_4 = 100

        # Create the student version
        self.stud_vers_4 = optimization.schedule_lr_decay(learning_rate_4, epoch_num_4, decay_rate_4, time_interval_4)

        # Load the reference
        self.exp_vers_4 = 0.24

    def test_updated_learning_rate_1(self):
        """ Test the updated learning rate 'updated_lr' for test 1. """

        stud_vers = self.stud_vers_1
        exp_vers = self.exp_vers_1

        self.assertAlmostEqual(stud_vers, exp_vers, delta=0.0001, msg="'updated_lr' is not correct!")

    def test_updated_learning_rate_2(self):
        """ Test the updated learning rate 'updated_lr' for test 2. """

        stud_vers = self.stud_vers_2
        exp_vers = self.exp_vers_2

        self.assertAlmostEqual(stud_vers, exp_vers, delta=0.0001, msg="'updated_lr' is not correct!")

    def test_updated_learning_rate_3(self):
        """ Test the updated learning rate 'updated_lr' for test 3. """

        stud_vers = self.stud_vers_3
        exp_vers = self.exp_vers_3

        self.assertAlmostEqual(stud_vers, exp_vers, delta=0.0001, msg="'updated_lr' is not correct!")

    def test_updated_learning_rate_4(self):
        """ Test the updated learning rate 'updated_lr' for test 4. """

        stud_vers = self.stud_vers_4
        exp_vers = self.exp_vers_4

        self.assertAlmostEqual(stud_vers, exp_vers, delta=0.0001, msg="'updated_lr' is not correct!")


if __name__ == '__main__':
    # Instantiate the command line parser
    parser = argparse.ArgumentParser()

    # Add the option to run only a specific test case
    parser.add_argument('--test_case', help='Name of the test case you want to run')

    # Read the command line parameters
    args = parser.parse_args()

    # Run only a single test class
    if args.test_case:
        test_class = eval(args.test_case)
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unittest.TextTestRunner().run(suite)

    # Run all test classes
    else:
        unittest.main(argv=[''], verbosity=1, exit=False)