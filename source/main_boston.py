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
    y_pred = u.predictions(x_train, W, 
                           logl_args[0], 
                           logl_args[1], 
                           logl_args[2])
    val = -u.mean_squared_error(y_pred, y_train)
    return val


#Store results
results_list = []
x_axis = range(1,20,1)
# Architecture for mlp
neurons_1 = x_train.shape[1] # inputs
output_neurons = 1

start = datetime.now()
for i  in x_axis:
    neurons_2 = i
    ndim_1 =  (neurons_1) * (neurons_2) + (neurons_2) * output_neurons
    nlive = ndim_1 * 100
    logl_args = [neurons_1, neurons_2, output_neurons]
    print("Units :::::", neurons_2, ":::::::::::::::::::\n")
    
    sampler = dynesty.NestedSampler(
                                   log_likelihood,
                                   prior_ptform,
                                   ndim = ndim_1,
                                   nlive = nlive,
                                   logl_args = logl_args,
                                   nparam = ndim_1,
                                   bound ='multi',
                                   sample ='hslice'
                                   ) 
    sampler.run_nested(dlogz=4000)
    res = sampler.results
    results_list.append(res)

end = datetime.now()

print("\n Time :::", end - start)

'''Plots '''
# train metrics
x = x_train
y = y_train
logZ, accuracy_list = u.return_results_regression(x, y, results_list, 
                                        x_axis, neurons_1, neurons_2, output_neurons)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Iris Dataset', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Log evidence')
plt.plot(x_axis, logZ, '-o')
plt.show()

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Iris Dataset -Train', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Accuracy')
plt.plot(x_axis, accuracy_list, '-o')
plt.show()

# test data set
x = x_test
y = y_test
logZ, accuracy_list = u.return_results_regression(x, y, results_list, 
                                        x_axis, neurons_1, neurons_2, output_neurons)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Iris Dataset - Test', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Accuracy')
plt.plot(x_axis, accuracy_list, '-o')
plt.show()

