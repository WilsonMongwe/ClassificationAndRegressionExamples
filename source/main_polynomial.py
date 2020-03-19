import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import numpy as np

    
def format_axis(ax):
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.grid()
    ax.legend()

# the data, split between train and test sets
# create synthetic data set
np.random.seed(2)
num_samples = 50
x_train = np.linspace(-2,2,num_samples)
x_test = np.linspace(-2,2,num_samples)+1/(2*num_samples)
y_train = 2 + 5*x_train**2 + 10*x_train**3+ np.random.normal(0, 2, num_samples)
y_test = 2 + 5*x_test**2 + 10*x_test**3+ np.random.normal(0, 2, num_samples)

# plot data set
fig, ax = plt.subplots()
dot_size = 10
ax.scatter(x_train, y_train, s=dot_size, label='train')
ax.scatter(x_test,y_test, s=dot_size, alpha=0.5, label='test')
format_axis(ax)
fig.savefig('results/poly_dataset.png')

def prior_ptform(uTheta):
    theta = norm.ppf(uTheta, loc = 0, scale =30)    
    return theta

def prediction_function(x_train, W, degree):
    y = 0
    for i in range(degree+1):
        y = y + W[i]*x_train**i
    return y

def error_function(y_pred, y_target):
    return u.mean_squared_error(y_pred, y_target)
    

def log_likelihood(W, *logl_args):
    y_pred = prediction_function(x_train, W, logl_args[0])
    val = -error_function(y_pred, y_train)
    return val


#Store results
results_list = [] # to store result of nested sampling
x_axis = range(1,7,1) # Number of hidden units

start = datetime.now()

for i  in x_axis:
    degree = i
    ndim_1 =  degree+1
    nlive = 100 * degree
    logl_args = [degree]
    
    print(" ")
    print("\n Degree of polynomial :::::", degree, ":::::::::::::::::::\n")
    random.seed(1)
    sampler = dynesty.NestedSampler(log_likelihood,
                                   prior_ptform,
                                   ndim = ndim_1,
                                   nlive = nlive,
                                   logl_args = logl_args,
                                   nparam = ndim_1,
                                   bound ='multi',
                                   sample ='hslice'
                                   #rstate = np.random.RandomState(1)
                                   ) 
    sampler.run_nested(maxiter =300)
    res = sampler.results
    results_list.append(res)

end = datetime.now()

print("\n Time :::", end - start)


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

'''Plots '''
# train metrics
x = x_train
y = y_train
logZ, mse_list_mode= return_results_regression_polynomial(x, y, results_list, x_axis)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Sythetic Dataset', fontsize=16)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('Log evidence')
plt.plot(x_axis, logZ, '-o', label="Log evidence")
#plt.plot(x_axis, logZ_LOWER, '-o', label="3 sd lower bound")
#plt.plot(x_axis, logZ_UPPER, '-o', label="3 sd upper bound")
plt.legend(loc=1)
plt.show()
fig.savefig('results/poly_log_evidence.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Sythetic Dataset -Train', fontsize=16)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('MSE')
plt.plot(x_axis, mse_list_mode, '-o', label="MSE from mode weights")
#plt.plot(x_axis, mse_list_mean, '-o', label="MSE from mean weights")
plt.legend(loc=2)
plt.show()
fig.savefig('results/poly_mse_train.png')


# test data set
x = x_test
y = y_test
logZ, mse_list_mode = return_results_regression_polynomial(x, y, results_list, x_axis)
plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Synthetic Dataset -Test', fontsize=16)
ax.set_xlabel('Degree of polynomial')
ax.set_ylabel('MSE')
plt.plot(x_axis, mse_list_mode, '-o', label="MSE from mode weights")
#plt.plot(x_axis, mse_list_mean, '-o', label="MSE from mean weights")
plt.legend(loc=2)
plt.show()
fig.savefig('results/poly_mse_test.png')