"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Shared functions of both the softmax regression network and the two layer neural network.

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

# Do not use packages that are not in standard distribution of python
import numpy as np


class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        Constructor.

        :param input_size: Size of the input.
        :param num_classes: Number of classes within the input.
        """

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        """
        Compute softmax scores given the raw output from the model.

        :param scores: raw scores from the model (N, num_classes)

        :return:
            prob: softmax probabilities (N, num_classes)
        """

        prob = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels.

        :param x_pred: Probabilities from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch

        :return: The computed Cross-Entropy Loss
        """

        loss = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        """
        Compute the accuracy of current batch.

        :param x_pred: Probabilities from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """

        acc = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        """
        Compute the sigmoid activation for the input.

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        """

        out = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Compute the sigmoid activation on the input                         #
        #############################################################################

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        """
        The analytical derivative of sigmoid function at x.

        :param x: Input data
        :return: The derivative of sigmoid function at x
        """

        ds = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input.

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        """

        out = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Compute the ReLU activation on the input                            #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self, X):
        """
        Compute the gradient ReLU activation for the input.

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        """

        out = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Compute the gradient of ReLU activation                             #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return out
