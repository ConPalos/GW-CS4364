# a nonlinear least squares model
import numpy as np
from costs import smape, mse_reg_manhattan, mse_reg_manhattan_gradient

class LinearRegressor:
    def __init__(self, learning_rate, threshold, regularizer):
        self.alpha = learning_rate
        self.threshold = threshold
        self.scores = []
        self.weight_hist = []
        self.regularizer = regularizer
    
    def evaluate(self, weights, features):
        '''Linear combination of weights and features'''
        return np.dot(features, weights)
    
    def __descend(self, data, answers):
        '''Gradient descent'''
        # gradient vector
        gradient = mse_reg_manhattan_gradient(answers, data, self.weights, self.regularizer)

        # current weights minus alpha times gradient
        return self.weights - self.alpha*gradient

    def fit(self, data, answers):
        '''
        Train the model on the given data
        @param data: a numpy array of train data with rows as entries
        @param answers: a numpy array of answers
        '''

        # start by generating the initial weights
        self.weights = np.zeros(data.shape[1])

        # descent the gradient
        self.weight_hist.append(self.weights)
        self.prev_weights = self.weights
        self.weights = self.__descend(data, answers)

        while self.__eval_convergence(data, answers) > self.threshold:
            self.weight_hist.append(self.weights)
            self.prev_weights = self.weights
            self.weights = self.__descend(data, answers)

        # upon convergence, return the scores
        return self.scores, self.weight_hist
        
    def __eval_convergence(self, data, answers):
        # evaluate on each set of weights
        current = self.evaluate(self.weights, data)
        prev = self.evaluate(self.prev_weights, data)

        # objective function for each
        curr_obj = mse_reg_manhattan(current, answers, self.regularizer, self.weights)
        prev_obj = mse_reg_manhattan(prev, answers, self.regularizer, self.prev_weights)

        # for plotting later
        self.scores.append(curr_obj)

        return abs(curr_obj - prev_obj)
    
    def test(self, data, answers):
        # make a prediction
        prediction = self.evaluate(self.weights, data)

        # check it against the answers
        score = mse_reg_manhattan(prediction, answers, self.regularizer, self.weights)

        # give a percentage score too
        percent = smape(prediction, answers)

        return score, percent