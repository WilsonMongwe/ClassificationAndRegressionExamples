import numpy as np
import sklearn
import pandas as pd
import scipy.special
from sklearn.model_selection import train_test_split
import tensorflow as tf
from csv import reader
import dynesty
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics


def softmax(x):
    return scipy.special.softmax(x, axis = 1)
 
def multi_cross_entropy(predictions, targets):   
    ce = -np.sum(targets*np.log(predictions+1e-9)) # No division by N
    return ce

def mean_squared_error(predictions, targets):
    mse = metrics.mean_squared_error(predictions, targets)
    return mse

def regression_neural_network(X, w_1, b_1, w_2):
    hidden_layer = np.tanh((X @ w_1) + b_1.T)
    y = hidden_layer @ w_2 # no bias required on this layer
    return y

def multi_class_neural_network(X, w_1, b_1, w_2, b_2):
    hidden_layer = np.tanh((X @ w_1) + b_1.T)
    y = softmax((hidden_layer @ w_2) + b_2.T)
    return y

def regression_predictions(X, W, input_neurons, hidden_neurons, output_neurons):
    param_1 = input_neurons * hidden_neurons
    bias_1  = hidden_neurons
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((input_neurons,hidden_neurons))
    b_1 = W[param_1:param_1+ bias_1]
    
    param_2 = hidden_neurons * output_neurons
    w_2 = W[param_1 + bias_1:param_1 + bias_1 + param_2]
    w_2 = w_2.reshape((hidden_neurons, output_neurons))

    predictions = regression_neural_network(X, w_1, b_1, w_2)
    
    return predictions


def multi_class_predictions(X, W, input_neurons, hidden_neurons, output_neurons): 
    param_1 = input_neurons * hidden_neurons
    bias_1  =  hidden_neurons
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((input_neurons,neurons_2))
    b_1 = W[param_1:param_1+ bias_1]
    
    param_2 = hidden_neurons * output_neurons
    bias_2 = output_neurons
    w_2 = W[param_1 + bias_1:param_1 + bias_1 + param_2]
    w_2 = w_2.reshape((hidden_neurons, output_neurons))
    b_2 = W[param_1 + bias_1 + param_2:param_1+ 
            bias_1 + param_2 + bias_2]

    predictions = multi_class_neural_network(X, w_1, b_1, w_2, b_2)
    
    return predictions


def return_results_regression(x, y, results_list, x_axis,
                              input_neurons, output_neurons):
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        hidden_neurons = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        #W, cov = dynesty.utils.mean_and_cov(samples, weights)
        W = samples[index_max_list[i-1]]
        predictions_list.append(
            regression_predictions(x, W, input_neurons,hidden_neurons, output_neurons))
    
    mse_list = []
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        mse_list.append(metrics.mean_squared_error(y, y_pred))
        
    return logZ, mse_list



def multi_return_results(x, y, results_list, x_axis, 
                         input_neurons, output_neurons):
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        hidden_neurons = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        #W, cov = dynesty.utils.mean_and_cov(samples, weights)
        W = samples[index_max_list[i-1]]
        predictions_list.append(predictions(x, W, 
                                            input_neurons, hidden_neurons, output_neurons))
    
    accuracy_list = []
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        accuracy_list.append(metrics.categorical_accuracy(y, y_pred))

    return logZ, accuracy_list


''' Loading in the data'''

def return_iris_processed_data():
      # Load digits dataset
    iris = datasets.load_iris()
    
    X = iris['data']
    y = iris['target']
    
    # One hot encoding of the output variable
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()
    
    # Scale features to have mean 0 and variance 1 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_boston_processed_data():
      # Load digits dataset
    X, Y = datasets.load_boston(return_X_y =True)
        
    # Scale features to have mean 0 and variance 1 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state = 1)
    
    return x_train, y_train, x_test, y_test
