"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Implementation of the Stochastic Gradient Descent (SGD) optimization method.

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

from ._base_optimizer import _BaseOptimizer


class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        """
        Update model weights based on gradients.

        :param model: The model to be updated.

        :return: None, but the model weights should be updated
        """

        self.apply_regularization(model)

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Update model weights based on the learning rate and gradients       #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
