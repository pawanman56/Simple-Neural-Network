"""
Otto-Friedrich University of Bamberg
Chair of Explainable Machine Learning (xAI)
Deep Learning Assignments

@description:
Base optimizer class for both networks.

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


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        """
        Apply L2 penalty to the model and update the gradient dictionary in the model.

        :param model: The model with gradients

        :return: None, but the gradient dictionary of the model should be updated
        """

        #############################################################################
        #                            START OF YOUR CODE                             #
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
