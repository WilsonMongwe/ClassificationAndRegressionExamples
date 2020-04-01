import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(1)

# the data, split between train and test sets
x_train, y_train, x_test, y_test = u.return_boston_processed_data()


def prior_ptform(uTheta):
    theta = norm.ppf(uTheta, loc = 0, scale = 1)    
    return theta



def log_likelihood(W, *logl_args):
    y_pred = u.regression_predictions(x_train, W, 
                           logl_args[0], 
                           logl_args[1], 
                           logl_args[2])
    val = -u.mean_squared_error(y_pred, y_train)
    return val


#Store results
results_list = [] # to store result of nested sampling
x_axis = range(1,21,1) # Number of hidden units
# Architecture for mlp
input_neurons = x_train.shape[1] # inputs
output_neurons = 1

start = datetime.now()
for i  in x_axis:
    hidden_neurons = i
    ndim_1 =  (input_neurons+1) * (hidden_neurons) + (hidden_neurons +1) * output_neurons
    nlive = 100 * ndim_1
    logl_args = [input_neurons, hidden_neurons, output_neurons]
    
    print(" ")
    print("\n Number of hidden neurons :::::", hidden_neurons, ":::::::::::::::::::\n")
   
    sampler = dynesty.NestedSampler(log_likelihood,
                                   prior_ptform,
                                   ndim = ndim_1,
                                   nlive = nlive,
                                   logl_args = logl_args,
                                   nparam = ndim_1,
                                   bound ='multi',
                                   sample ='hslice'
                                   ) 
    sampler.run_nested(maxiter = 5000)
    res = sampler.results
    results_list.append(res)

end = datetime.now()

print("\n Time :::", end - start)

'''Plots '''
# train metrics
x = x_train
y = y_train
logZ, mse_list_mode = u.return_results_regression(x, y, results_list, 
                                        x_axis, input_neurons, output_neurons)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Boston Dataset', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Log evidence')
plt.plot(x_axis, logZ, '-o', label="Log evidence")
#plt.plot(x_axis, logZ_LOWER, '-o', label="3 sd lower bound")
#plt.plot(x_axis, logZ_UPPER, '-o', label="3 sd upper bound")
plt.legend(loc=1)
plt.show()
fig.savefig('results/boston_log_evidence.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Boston Dataset -Train', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('MSE')
plt.plot(x_axis, mse_list_mode, '-o', label="MSE from mode weights")
#plt.plot(x_axis, mse_list_mean, '-o', label="MSE from mean weights")
plt.legend(loc=3)
plt.show()
fig.savefig('results/boston_mse_train.png')
# test data set
x = x_test
y = y_test
logZ, mse_list_mode  = u.return_results_regression(x, y, results_list, 
                                        x_axis, input_neurons, output_neurons)
plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Boston Dataset -Test', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('MSE')
plt.plot(x_axis, mse_list_mode, '-o', label="MSE from mode weights")
#plt.plot(x_axis, mse_list_mean, '-o', label="MSE from mean weights")
plt.legend(loc=3)
plt.show()
fig.savefig('results/boston_mse_testpng')