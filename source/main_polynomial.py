import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import metrics
import csv  
import pandas as pd

# Create synthetic data set
random.seed(1)
num_samples = 50
x_train = np.linspace(-2,2,num_samples)
x_test = np.linspace(-2,2,num_samples)+1/(2*num_samples)
y_train = 2 + 5*x_train**2 + 7*x_train**3+ 10*x_train**6 + np.random.normal(0, 3, num_samples)
y_test = 2 + 5*x_test**2 + 7*x_test**3+ 10*x_train**6 + np.random.normal(0, 3, num_samples)

synthetic_data_file_name = r'synthetic6.csv'
logz_file_name = r'logz6.csv'
mse_train_file_name = r'mse_train6.csv'
mse_test_file_name = r'mse_test6.csv'

#Store results
x_axis = range(1, 11, 1) # Degree of polynomial
sims = range(1, 31, 1) # How many times to runj the simulations

with open(synthetic_data_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(x_train)

with open(synthetic_data_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(y_train)

with open(synthetic_data_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(x_test)

with open(synthetic_data_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(y_test)

# prior
def prior_ptform(uTheta):
    theta = norm.ppf(uTheta, loc = 0, scale = 30)    
    return theta

def prediction_function(x_train, W, degree):
    y = 0
    for i in range(degree+1):
        y = y + W[i]*x_train**i
    return y

def error_function(y_pred, y_target):
    return metrics.mean_squared_error(y_pred, y_target)
  
# likelihood
def log_likelihood(W, *logl_args):
    y_pred = prediction_function(x_train, W, logl_args[0])
    val = -error_function(y_pred, y_train)
    return val

# regression results calculation
def return_results_regression_polynomial(x, y, results_list, x_axis):
    
    logZ =[]
    for i in results_list:
        logZ.append(i.logz[-1])

    samples_list = []
    weights_list = []
    index_max_list = []
    predictions_list_mode = []
    
    for i in results_list:
        samples, weights = i.samples, np.exp(i.logwt - i.logz[-1])
        samples_list.append(samples)
        weights_list.append(weights)
        index_max_list.append(np.argmax(weights))
 
    for i  in x_axis: 
        degree = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        W_mode = samples[index_max_list[i-1]]
        predictions_list_mode.append(
            prediction_function(x, W_mode,degree))
    
    mse_list_mode = []
    for i  in x_axis:
        y_pred_mode = predictions_list_mode[i-1]
        mse_list_mode.append(error_function(y, y_pred_mode))
        
    return logZ, mse_list_mode



with open(logz_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis)
with open(mse_train_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis)
with open(mse_test_file_name, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(x_axis)


start = datetime.now()
    
for s  in sims:
    print(" ")
    print("\n SIMULATION NUMBER :::::::::::::::::::::", s, "::::::::::::::::::::::::\n")
    results_list = [] # to store result of nested sampling
    
    for i  in x_axis:
        degree = i
        ndim_1 =  degree+1
        nlive = 100 * degree
        logl_args = [degree]
        
        print(" ")
        print("\n Degree of polynomial :::::", degree, ":::::::::::::::::::\n")
        
        sampler = dynesty.NestedSampler(log_likelihood,
                                       prior_ptform,
                                       ndim = ndim_1,
                                       nlive = nlive,
                                       logl_args = logl_args
                                       ) 
        sampler.run_nested(maxiter = 10000)
        res = sampler.results
        results_list.append(res)
        
    # train data
    x = x_train
    y = y_train
    logZ_train, mse_list_mode_train= return_results_regression_polynomial(x, y, results_list, x_axis)
    
    # test data set
    x = x_test
    y = y_test
    logZ_not_used, mse_list_mode_test = return_results_regression_polynomial(x, y, results_list, x_axis)
    
    with open(logz_file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(logZ_train)
    with open(mse_train_file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(mse_list_mode_train)
    with open(mse_test_file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(mse_list_mode_test)

end = datetime.now()

print("\n Time :::", end - start)


'''Plots '''
#read in data stored in the files
synthetic_data = pd.read_csv("synthetic6.csv", header=None) 
data_logz = pd.read_csv("logz6.csv") 
data_mse_train = pd.read_csv("mse_train6.csv") 
data_mse_test = pd.read_csv("mse_test6.csv") 

x_train = synthetic_data.iloc[0]
y_train = synthetic_data.iloc[1]
x_test = synthetic_data.iloc[2]
y_test = synthetic_data.iloc[3]

logz_train_avarage = data_logz.mean()
mse_train_avarage = data_mse_train.mean()
mse_test_avarage = data_mse_test.mean()

with open(r'logz6_mean.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(logz_train_avarage)
    
with open(r'mse_train6_mean.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(mse_train_avarage)

with open(r'mse_test6_mean.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(mse_test_avarage)

# plot data set
start = 2
end = 11   
x_axis = range(start, end, 1)     
    
def format_axis(ax):
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.grid()
    ax.legend()
    
fig, ax = plt.subplots()
dot_size = 10
ax.scatter(x_train, y_train, s = dot_size, label = 'Train')
ax.scatter(x_test, y_test, s = dot_size, alpha = 0.5, label='Test')
format_axis(ax)
fig.savefig('results/poly_dataset.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Train Dataset', fontsize=16)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('Average log evidence')
plt.plot(x_axis, logz_train_avarage[start-1:end-1], '-o', label="Log evidence")
plt.legend(loc=4)
plt.show()
fig.savefig('results/poly_log_evidence.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Train Dataset', fontsize=16)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('Average MSE')
plt.plot(x_axis, mse_train_avarage[start-1:end-1], '-o', label="MSE from mode weights")
plt.legend(loc=1)
plt.show()
fig.savefig('results/poly_mse_train.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Test Dataset', fontsize=16)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('Average MSE')
plt.plot(x_axis, mse_test_avarage[start-1:end-1], '-o', label="MSE from mode weights")
plt.legend(loc=1)
plt.show()
fig.savefig('results/poly_mse_test.png')



