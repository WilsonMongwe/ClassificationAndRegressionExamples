import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


# Create synthetic data set
random.seed(1)
num_samples = 50
x_train = np.linspace(-2,2,num_samples)
x_test = np.linspace(-2,2,num_samples)+1/(2*num_samples)
y_train = 2 + 5*x_train**2 + 7*x_train**3+ np.random.normal(0, 3, num_samples)
y_test = 2 + 5*x_test**2 + 7*x_test**3+ np.random.normal(0, 3, num_samples)

x_train = np.array(x_train).reshape((num_samples, 1))
y_train = np.array(y_train).reshape((num_samples, 1))
x_test = np.array(x_test).reshape((num_samples, 1))
y_test = np.array(y_test).reshape((num_samples, 1))

# scaling the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

def prior_ptform(uTheta):
    theta = norm.ppf(uTheta, loc = 0, scale = 31.2500)    
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
x_axis = range(1,11,1) # Number of hidden units
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
                                   logl_args = logl_args
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
logZ_train, mse_list_mode = u.return_results_regression(x, y, results_list, 
                                        x_axis, input_neurons, output_neurons)

# test data set
x = x_test
y = y_test
logZ, mse_list_mode  = u.return_results_regression(x, y, results_list, 
                                        x_axis, input_neurons, output_neurons)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Train Dataset', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Log evidence')
plt.plot(x_axis, logZ_train, '-o', label="Log evidence")
plt.legend(loc=1)
plt.show()
fig.savefig('results/boston_log_evidence.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle(' Train Dataset', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('MSE')
plt.plot(x_axis, mse_list_mode, '-o', label="MSE from mode weights")
plt.legend(loc=3)
plt.show()
fig.savefig('results/boston_mse_train.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Test Dataset', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('MSE')
plt.plot(x_axis, mse_list_mode, '-o', label="MSE from mode weights")
plt.legend(loc=3)
plt.show()
fig.savefig('results/boston_mse_testpng')