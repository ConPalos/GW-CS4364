import numpy as np

def norm(vector):
    '''The L2 norm of a vector'''
    return np.sqrt(np.multiply(vector, vector).sum())

def mse(output, answers):
    diff = answers - output
    error = 0.5*(diff.T @ diff)/answers.shape[0]
    return error

def mse_gradient(features, answers, weights):
    by_weight = (features.T @ features) @ weights
    by_answers = features.T @ answers
    return by_weight - by_answers

def mse_reg(output, answers, regularizer, weights):
    mse_term = mse(output, answers)
    reg_term = regularizer*0.5*(norm(weights)**2)
    return np.squeeze(mse_term + reg_term)

def mse_reg_gradient(answers, features, weights, regularizer):
    mse_grad = mse_gradient(features, answers, weights)
    reg_grad = regularizer*weights
    return mse_grad + reg_grad

def mse_reg_manhattan(output, answers, regularizer, weights):
    mse_term = mse(output, answers)
    reg_term = regularizer*np.sum(np.squeeze(weights))
    return np.squeeze(mse_term + reg_term)

def mse_reg_manhattan_gradient(answers, features, weights, regularizer):
    mse_grad = mse_gradient(features, answers, weights)
    reg_grad = regularizer
    return mse_grad + reg_grad

def smape(prediction, answers):
    # absolute difference
    diff = np.squeeze(abs(prediction - answers))

    # sum of the absolute values
    sums = np.squeeze(abs(prediction) + abs(answers))

    n = len(answers)

    error = (2*diff)/(n*sums)
    return error