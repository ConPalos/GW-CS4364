import numpy as np

class RidgeRegressor:
    def __init__(self, regularizer):
        self.regularizer = regularizer

    def __cost(self, output, answers):
        '''Mean squared error'''
        # start with the difference of matrices to find the absolute error
        error = answers - output

        # multiply the matrix by itself
        error = error.T @ error

        # ok now that that's done, regularize!
        norm = self.__norm(self.weights)
        error += self.regularizer*(norm**2)

        # and divide by 2
        error /= 2

        return error
    
    def __norm(self, vector):
        n = np.multiply(vector, vector)
        n = sum(sum(n))
        n = n**0.5
        return n

    def evaluate(self, weights, features):
        '''Linear combination of weights and features'''
        return features @ weights

    def fit(self, data, answers):
        '''
        Train the model on the given data
        @param data: a numpy array of train data with rows as entries
        @param answers: a numpy array of answers
        '''
        # the normal equations provide a closed form for ridge regression
        identity = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            identity[i][i] = 1

        # the answers need to be a vertical vector (otherwise the matrix multiplication fails)
        answers = np.array([answers]).T
        
        self.weights = np.linalg.inv((data.T @ data) + self.regularizer*identity) @ data.T @ answers

        # upon convergence, return the score
        return self.__cost(self.evaluate(self.weights, data), answers)
        
    def test(self, data, answers):
        # make a prediction
        prediction = self.evaluate(self.weights, data)

        # check it against the answers
        score = self.__cost(prediction, answers)

        return score