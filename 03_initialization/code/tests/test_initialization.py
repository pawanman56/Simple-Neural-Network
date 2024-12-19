#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the 'initialization.ipynb'-jupyter notebook.
That notebook lets you analyze different weight initialization methods on some sample data.
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
import initialization


class TestDataGeneration(unittest.TestCase):
    """
    The class contains all test cases for task "2 - Dataset Generation".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        n_samples_train, n_samples_test = 10, 5
        noise_train, noise_test = .05, .05
        seed_train, seed_test = 1, 2

        # Create the student version
        stud_o1, stud_o2, stud_o3, stud_o4 = initialization.generate_dataset(n_samples_train, n_samples_test,
                                                                             noise_train, noise_test, seed_train,
                                                                             seed_test)

        self.stud_vers = {"train_X": stud_o1, "train_Y": stud_o2, "test_X": stud_o3, "test_Y": stud_o4}

        # Load the references
        self.exp_vers = {"train_X": np.array([[-0.72987938, 0.53552791], [0.26009327, -0.85881494],
                                              [0.17182321, 0.74508061], [0.35192594, -0.94748937],
                                              [0.88538454, -0.07077762], [-0.76472552, -0.55619016],
                                              [0.31103001, 0.90590037], [-0.63844746, 0.47994975],
                                              [-0.67397605, -0.43136064], [0.80856692, -0.02214257]]),
                         "train_Y": np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
                         "test_X": np.array([[0.6931902, 0.08201354], [-0.48967178, -0.73490769],
                                             [-0.97485593, -0.0622644], [-0.45289761, 0.64736994],
                                             [1.0275727, 0.1146104]]),
                         "test_Y": np.array([1, 1, 0, 1, 0])
                         }

    def test_train_X(self):
        """ Test generating the training samples. """

        stud_vers = self.stud_vers["train_X"]
        exp_vers = self.exp_vers["train_X"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of train_X is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of train_X is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "train_X is not correct!")

    def test_train_Y(self):
        """ Test generating the training labels. """

        stud_vers = self.stud_vers["train_Y"]
        exp_vers = self.exp_vers["train_Y"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of train_Y is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of train_Y is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "train_Y is not correct!")

    def test_test_X(self):
        """ Test generating the test samples. """

        stud_vers = self.stud_vers["test_X"]
        exp_vers = self.exp_vers["test_X"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of test_X is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of test_X is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "test_X is not correct!")

    def test_test_Y(self):
        """ Test generating the test labels. """

        stud_vers = self.stud_vers["test_Y"]
        exp_vers = self.exp_vers["test_Y"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of test_Y is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of test_Y is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "test_Y is not correct!")


class TestPreprocessing(unittest.TestCase):
    """
    The class contains all test cases for task "3 - Preprocessing".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        train_X = np.array([[-0.72987938, 0.53552791], [0.26009327, -0.85881494], [0.17182321, 0.74508061],
                            [0.35192594, -0.94748937], [0.88538454, -0.07077762], [-0.76472552, -0.55619016],
                            [0.31103001, 0.90590037], [-0.63844746, 0.47994975], [-0.67397605, -0.43136064],
                            [0.80856692, -0.02214257]])
        train_Y = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        test_X = np.array([[0.6931902, 0.08201354], [-0.48967178, -0.73490769], [-0.97485593, -0.0622644],
                           [-0.45289761, 0.64736994], [1.0275727, 0.1146104]])
        test_Y = np.array([1, 1, 0, 1, 0])

        # Create the student version
        stud_o1, stud_o2, stud_o3, stud_o4 = initialization.preprocess(train_X, train_Y, test_X, test_Y)

        self.stud_vers = {"train_X": stud_o1, "train_Y": stud_o2, "test_X": stud_o3, "test_Y": stud_o4}

        # Load the references
        self.exp_vers = {"train_X": np.array([[-0.72987938, 0.26009327, 0.17182321, 0.35192594, 0.88538454, -0.76472552,
                                               0.31103001, -0.63844746, -0.67397605, 0.80856692],
                                              [0.53552791, -0.85881494, 0.74508061, -0.94748937, -0.07077762,
                                               -0.55619016, 0.90590037, 0.47994975, -0.43136064, -0.02214257]]),
                         "train_Y": np.array([[0, 1, 1, 0, 0, 0, 0, 1, 1, 1]]),
                         "test_X": np.array([[0.6931902, -0.48967178, -0.97485593, -0.45289761, 1.0275727],
                                             [0.08201354, -0.73490769, -0.0622644, 0.64736994, 0.1146104]]),
                         "test_Y": np.array([[1, 1, 0, 1, 0]])
                         }

    def test_preprocess_train_X(self):
        """ Test preprocessing the training samples. """

        stud_vers = self.stud_vers["train_X"]
        exp_vers = self.exp_vers["train_X"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of preprocessed_train_X is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of preprocessed_train_X is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "preprocessed_train_X is not correct!")

    def test_preprocess_train_Y(self):
        """ Test preprocessing the training labels. """

        stud_vers = self.stud_vers["train_Y"]
        exp_vers = self.exp_vers["train_Y"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of preprocessed_train_Y is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of preprocessed_train_Y is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "preprocessed_train_Y is not correct!")

    def test_preprocess_test_X(self):
        """ Test preprocessing the test samples. """

        stud_vers = self.stud_vers["test_X"]
        exp_vers = self.exp_vers["test_X"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of preprocessed_test_X is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of preprocessed_test_X is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "preprocessed_test_X is not correct!")

    def test_preprocess_test_Y(self):
        """ Test preprocessing the test labels. """

        stud_vers = self.stud_vers["test_Y"]
        exp_vers = self.exp_vers["test_Y"]

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of preprocessed_test_Y is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of preprocessed_test_Y is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "preprocessed_test_Y is not correct!")


class TestZeroInitialization(unittest.TestCase):
    """
    The class contains all test cases for task "4.1 - Zero Initialization".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        layer_dims = [3, 2, 1]

        # Create the student version
        self.stud_vers = initialization.initialize_parameters_zeros(layer_dims)

        # Load the references
        self.exp_vers = {'W1': np.array([[0., 0., 0.],
                                         [0., 0., 0.]]),
                         'b1': np.array([[0.],
                                         [0.]]),
                         'W2': np.array([[0., 0.]]),
                         'b2': np.array([[0.]])
                         }

    def test_parameters(self):
        """ Test the parameter dictionary. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'parameters' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'parameters' is not correct!")

    def test_W1(self):
        """ Test the weights of layer 1. """

        self.assertIn('W1', self.stud_vers.keys(), "Weights for layer 1 are missing!")

        stud_vers = self.stud_vers['W1']
        exp_vers = self.exp_vers['W1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W1' is not correct!")

    def test_b1(self):
        """ Test the bias of layer 1. """

        self.assertIn('b1', self.stud_vers.keys(), "Bias for layer 1 is missing!")

        stud_vers = self.stud_vers['b1']
        exp_vers = self.exp_vers['b1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b1' is not correct!")

    def test_W2(self):
        """ Test the weights of layer 2. """

        self.assertIn('W2', self.stud_vers.keys(), "Weights for layer 2 are missing!")

        stud_vers = self.stud_vers['W2']
        exp_vers = self.exp_vers['W2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W2' is not correct!")

    def test_b2(self):
        """ Test the bias of layer 2. """

        self.assertIn('b2', self.stud_vers.keys(), "Bias for layer 2 is missing!")

        stud_vers = self.stud_vers['b2']
        exp_vers = self.exp_vers['b2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b2' is not correct!")


class TestRandomInitialization(unittest.TestCase):
    """
    The class contains all test cases for task "4.2 - Random Initialization".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        layer_dims = [3, 2, 1]
        seed = 3

        # Create the student version
        self.stud_vers = initialization.initialize_parameters_random(layer_dims, seed)

        # Load the references
        self.exp_vers = {'W1': np.array([[ 17.88628473,   4.36509851,   0.96497468],
                                         [-18.63492703,  -2.77388203,  -3.54758979]]),
                         'b1': np.array([[0.],
                                         [0.]]),
                         'W2': np.array([[-0.82741481, -6.27000677]]),
                         'b2': np.array([[0.]])}

    def test_parameters(self):
        """ Test the parameter dictionary. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'parameters' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'parameters' is not correct!")

    def test_W1(self):
        """ Test the weights of layer 1. """

        self.assertIn('W1', self.stud_vers.keys(), "Weights for layer 1 are missing!")

        stud_vers = self.stud_vers['W1']
        exp_vers = self.exp_vers['W1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W1' is not correct!")

    def test_b1(self):
        """ Test the bias of layer 1. """

        self.assertIn('b1', self.stud_vers.keys(), "Bias for layer 1 is missing!")

        stud_vers = self.stud_vers['b1']
        exp_vers = self.exp_vers['b1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b1' is not correct!")

    def test_W2(self):
        """ Test the weights of layer 2. """

        self.assertIn('W2', self.stud_vers.keys(), "Weights for layer 2 are missing!")

        stud_vers = self.stud_vers['W2']
        exp_vers = self.exp_vers['W2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W2' is not correct!")

    def test_b2(self):
        """ Test the bias of layer 2. """

        self.assertIn('b2', self.stud_vers.keys(), "Bias for layer 2 is missing!")

        stud_vers = self.stud_vers['b2']
        exp_vers = self.exp_vers['b2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b2' is not correct!")


class TestHeInitialization(unittest.TestCase):
    """
    The class contains all test cases for task "4.3 - He Initialization".
    """

    def setUp(self):
        """ Initialize the tests. """

        # Create the parameters
        layer_dims = [3, 1, 2]
        seed = 3

        # Create the student version
        self.stud_vers = initialization.initialize_parameters_he(layer_dims, seed)

        # Load the references
        self.exp_vers = {'W1': np.array([[1.46040903, 0.3564088, 0.07878985]]),
                         'b1': np.array([[0.]]),
                         'W2': np.array([[-2.63537665], [-0.39228616]]),
                         'b2': np.array([[0.], [0.]])}

    def test_parameters(self):
        """ Test the parameter dictionary. """

        stud_vers = self.stud_vers
        exp_vers = self.exp_vers

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'parameters' is not correct!")
        self.assertEqual(len(stud_vers.keys()), len(exp_vers.keys()), "Number of entries of 'parameters' is not correct!")

    def test_W1(self):
        """ Test the weights of layer 1. """

        self.assertIn('W1', self.stud_vers.keys(), "Weights for layer 1 are missing!")

        stud_vers = self.stud_vers['W1']
        exp_vers = self.exp_vers['W1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W1' is not correct!")

    def test_b1(self):
        """ Test the bias of layer 1. """

        self.assertIn('b1', self.stud_vers.keys(), "Bias for layer 1 is missing!")

        stud_vers = self.stud_vers['b1']
        exp_vers = self.exp_vers['b1']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b1' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b1' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b1' is not correct!")

    def test_W2(self):
        """ Test the weights of layer 2. """

        self.assertIn('W2', self.stud_vers.keys(), "Weights for layer 2 are missing!")

        stud_vers = self.stud_vers['W2']
        exp_vers = self.exp_vers['W2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'W2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'W2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'W2' is not correct!")

    def test_b2(self):
        """ Test the bias of layer 2. """

        self.assertIn('b2', self.stud_vers.keys(), "Bias for layer 2 is missing!")

        stud_vers = self.stud_vers['b2']
        exp_vers = self.exp_vers['b2']

        self.assertEqual(type(stud_vers), type(exp_vers), "Type of 'b2' is not correct!")
        self.assertEqual(stud_vers.shape, exp_vers.shape, "Shape of 'b2' is not correct!")
        self.assertTrue(np.allclose(stud_vers, exp_vers, atol=0.0001), "'b2' is not correct!")


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