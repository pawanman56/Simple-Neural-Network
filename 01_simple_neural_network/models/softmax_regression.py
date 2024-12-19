"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Implementation of a simple softmax regression network.

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

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax

        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """

        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        """
        Initialize weights of the single layer regression network. No bias term included.

        :return: None;
            self.weights is filled based on method
                - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        """
        np.random.seed(1024)

        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)

        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """

        loss = None
        gradient = None
        accuracy = None

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #       1.1) Computed the net (weighted) input.                             #
        #       1.2) Apply the ReLU-activation function.                            #
        #       1.3) Apply the Softmax function                                     #
        #       1.4) Compute the Cross-Entropy loss                                 #
        #       1.5) Compute the model's accuracy                                   #
        #                                                                           #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    2) Implement the backward process:                                     #
        #       2.1) Calculate the derivative of the loss with respect to the       #
        #            softmax-function.                                              #
        #       2.2) Calculate the derivative of the ReLU activation function with  #
        #            respect to the net (weighted) input.                           #
        #       2.3) Apply the chain rule to calculate the derivative of the loss   #
        #            with respect to the weights.                                   #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


