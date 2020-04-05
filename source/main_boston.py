import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import csv  
import pandas as pd

random.seed(1)

# the data, split between train and test sets
x_train, y_train, x_test, y_test = u.return_boston_processed_data()

logz_file_name = r'logz6.csv'
mse_train_file_name = r'mse_train6.csv'
mse_test_file_name = r'mse_test6.csv'

#Store results
x_axis = range(1, 31, 1) # Degree of polynomial
sims = range(1, 31, 1) # How many times to runj the simulations
# Architecture for mlp
input_neurons = x_train.shape[1] # inputs
output_neurons = 1

def prior_ptform(uTheta):
    theta = norm.ppf(uTheta, loc = 0, scale = 5)    
    return theta

def log_likelihood(W, *logl_args):
    y_pred = u.regression_predictions(x_train, W, 
                           logl_args[0], 
                           logl_args[1], 
                           logl_args[2])
    val = -u.mean_squared_error(y_pred, y_train)
    return val

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
        
    # train metrics
    x = x_train
    y = y_train
    logZ_train, mse_list_mode_train = u.return_results_regression(x, y, results_list, 
                                            x_axis, input_neurons, output_neurons)
    
    # test data set
    x = x_test
    y = y_test
    logZ_test, mse_list_mode_test  = u.return_results_regression(x, y, results_list, 
                                            x_axis, input_neurons, output_neurons)
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
data_logz = pd.read_csv("logz6.csv") 
data_mse_train = pd.read_csv("mse_train6.csv") 
data_mse_test = pd.read_csv("mse_test6.csv") 

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
start = 1
end = 30  
x_axis = range(start, end, 1)     
    
plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Train Dataset', fontsize=16)
ax.set_xlabel('Number of hidden neurons')
ax.set_ylabel('Average log evidence')
plt.plot(x_axis, logz_train_avarage[start-1:end-1], '-o', label="Log evidence")
plt.legend(loc=4)
plt.show()
fig.savefig('results/boston_log_evidence.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Train Dataset', fontsize=16)
ax.set_xlabel('Number of hidden neurons')
ax.set_ylabel('Average MSE')
plt.plot(x_axis, mse_train_avarage[start-1:end-1], '-o', label="MSE from mode weights")
plt.legend(loc=1)
plt.show()
fig.savefig('results/boston_mse_train.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Test Dataset', fontsize=16)
ax.set_xlabel('Number of hidden neurons')
ax.set_ylabel('Average MSE')
plt.plot(x_axis, mse_test_avarage[start-1:end-1], '-o', label="MSE from mode weights")
plt.legend(loc=1)
plt.show()
fig.savefig('results/boston_mse_test.png')

print(logz_train_avarage)
print(mse_test_avarage)

