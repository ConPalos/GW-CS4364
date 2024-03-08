import numpy as np
from costs import mse_reg_manhattan, mse_reg_manhattan_gradient, norm
import matplotlib.pyplot as plt

class NLSModel:
    def __init__(self, learning_rate, threshold, regularizer, degree):
        self.alpha = learning_rate
        self.threshold = threshold
        self.scores = []
        self.weight_hist = []
        self.grad_hist = []
        self.regularizer = regularizer
        self.degree = degree

    def get_scores(self):
        return self.scores
    
    def get_weights(self):
        w = np.squeeze(np.array(self.weight_hist))
        return w

    # fit the model
    def fit(self, data, answers):
        # expand the data into a polynomial of a given degree (each row is now much longer)
        features = self.__psi(data)

        # answers needs to be a matrix, not a vector
        if len(answers.shape) == 1:
            answers = np.expand_dims(answers, 1)

        # create the weights
        _, f = features.shape
        self.weights = np.random.rand(f, 1)

        # record history
        self.weight_hist.append(self.weights)
        self.weights = self.__descend(features, answers)

        # measure convergence
        while self.__eval_convergence(features, answers) > self.threshold:
            self.weight_hist.append(self.weights)
            self.weights = self.__descend(features, answers)

        # upon convergence, return
        return
    
    # descend the gradient
    def __descend(self, features, answers):
        # get the gradient
        gradient = mse_reg_manhattan_gradient(answers, features, self.weights, self.regularizer)
        self.grad_hist.append(gradient)

        # step in the direction of the gradient
        return self.weights - self.alpha*gradient

    def __eval_convergence(self, features, answers):
        # check the error with the current weight set
        current_output = self.__predict(self.weights, features)
        current_error = mse_reg_manhattan(current_output, answers, self.regularizer, self.weights)
        
        self.scores.append(np.squeeze(current_error))

        # check the error with the previous weight set
        previous_output = self.__predict(self.weight_hist[-1], features)
        previous_error = mse_reg_manhattan(previous_output, answers, self.regularizer, self.weight_hist[-1])

        return abs(current_error - previous_error)
    
    def __predict(self, weights, features):
        # linear combo is backwards
        return features @ weights
    
    def predict(self, weights, features):
        # start with psi
        expanded = self.__psi(features)
        return expanded @ weights
    
    def __psi(self, data):
        # unusually, this will have no convolution
        # meaning that variables will stay independent
        # but i am still modeling a polynomial of the given degree
        # i think i'm using the term "convolution" right here
        # anyway, the algorithm is to just keep concatting features
        output = np.copy(data)
        to_add = np.copy(data)
        for exponent in range(2, self.degree + 1):
            # each value of exponent represents what's being concatted
            to_add = np.multiply(to_add, data) # squared, cubed, etc
            output = np.concatenate([output, to_add], axis=1)

        # at this point, the feature space is immense
        # there is a term for each variable up to the given exponent
        return output