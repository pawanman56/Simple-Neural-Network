"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
This script was written as a test module for the implementation of the data loading and preprocessing part.
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
from utils import load_mnist_trainval, load_mnist_test, generate_batched_data

class TestLoading(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def test_load_mnist(self):
        """ Test loading the dataset. """

        train_data, train_label, val_data, val_label = load_mnist_trainval()
        self.assertEqual(len(train_data), len(train_label))
        self.assertEqual(len(val_data), len(val_label))
        self.assertEqual(len(train_data), 4*len(val_data))
        for img in train_data:
            self.assertIsInstance(img, list)
            self.assertEqual(len(img), 784)
        for img in val_data:
            self.assertIsInstance(img, list)
            self.assertEqual(len(img), 784)
        for t in train_label:
            self.assertIsInstance(t, int)
        for t in val_label:
            self.assertIsInstance(t, int)

    def test_generate_batch(self):
        """ Test generating data batches. """

        train_data, train_label, val_data, val_label = load_mnist_trainval()
        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=128, shuffle=True, seed=1024)
        for i, b in enumerate(batched_train_data[:-1]):
            self.assertEqual(len(b), 128)
            self.assertEqual(len(batched_train_label[i]), 128)

