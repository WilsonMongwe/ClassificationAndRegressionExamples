import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import csv  
import pandas as pd
import math

# the data, split between train and test sets
random.seed(1)
x_train, y_train, x_test, y_test = u.return_concrete_processed_data()


# write out train data
diabetes_train = r'concrete_data_train.csv'
df_train = pd.DataFrame(data=x_train)
df_train[8] = y_train 
df_train.to_csv(diabetes_train, index =False, header =False)

# write out test data
diabetes_test = r'concrete_data_test.csv'
df_test = pd.DataFrame(data=x_test)
df_test[8] = y_test
df_test.to_csv(diabetes_test, index =False, header =False)


logz_file_name = r'logz_concrete.csv'

#Store results
x_axis = range(1, 31, 1) # Degree of polynomial
sims = range(1, 31, 1) # How many times to runj the simulations
# Architecture for mlp
input_neurons = x_train.shape[1] # inputs
output_neurons = 1

# Read in alpha and beta from GA results
alpha = pd.read_excel("alpha_concrete_mean.xlsx").values
beta = pd.read_excel("beta_concrete_mean.xlsx").values

alpha = alpha.reshape(30,1)
beta = beta.reshape(30,1)

def prior_ptform(uTheta,*ptform_args):
    alpha_val = ptform_args[0]
    theta = norm.ppf(uTheta, loc = 0, scale = 1/alpha_val)
    return theta

def log_likelihood(W, *logl_args):
    y_pred = u.regression_predictions(x_train, W, 
                           logl_args[0], 
                           logl_args[1], 
                           logl_args[2])
    mse = -u.mean_squared_error(y_pred, y_train)
    beta_val = logl_args[3]
    return mse * beta_val

start = datetime.now()
    
for s  in sims:
    print(" ")
    print("\n SIMULATION NUMBER :::::::::::::::::::::", s, "::::::::::::::::::::::::\n")
    results_list = [] # to store result of nested sampling
    
    for i  in x_axis:
        hidden_neurons = i
        ndim_1 =  (input_neurons+1) * (hidden_neurons) + (hidden_neurons + 1) * output_neurons
        nlive = 100 * ndim_1
        logl_args = [input_neurons, hidden_neurons, output_neurons, beta[29][0]]
        ptform_args = [alpha[29][0]]
        print(" ")
        print("\n Number of hidden neurons :::::", hidden_neurons, ":::::::::::::::::::\n")
       
        sampler = dynesty.NestedSampler(log_likelihood,
                                       prior_ptform,
                                       ndim = ndim_1,
                                       nlive = nlive,
                                       logl_args = logl_args,
                                       ptform_args = ptform_args
                                       )
        sampler.run_nested(maxiter = 200)
        res = sampler.results
        results_list.append(res)
        
    # train metrics
    x = x_train
    y = y_train
    logZ_train = u.return_results_regression_simple(results_list)
    
    with open(logz_file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(logZ_train)

end = datetime.now()

print("\n Time :::", end - start)

'''Plots '''
#read in data stored in the files 
data_logz = pd.read_csv("logz_concrete.csv", header=None) 
logz_train_avarage = data_logz.mean()

with open(r'logz_concrete_mean.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(logz_train_avarage)
    
# plot data set
start = 1
end = 30  
x_axis = range(start, end, 1)     
    
plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Training Dataset', fontsize = 16)
ax.set_xlabel('Number of hidden neurons')
ax.set_ylabel('Average log evidence')
plt.plot(x_axis, logz_train_avarage[start-1:end-1], '-o')
plt.show()
fig.savefig('results/concrete_log_evidence.png')

print(logz_train_avarage)


