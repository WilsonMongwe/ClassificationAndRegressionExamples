import utilities as u
import dynesty
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor


random.seed(1)

# the data, split between train and test sets
x_train, y_train, x_test, y_test = u.return_taiwan_processed_data()


def prior_ptform(uTheta):
    theta = norm.ppf(uTheta, loc = 0, scale = 1)    
    return theta

def log_likelihood(W, *logl_args):
    y_pred = u.single_class_predictions(x_train, W, 
                           logl_args[0], 
                           logl_args[1], 
                           logl_args[2])
    val = -u.single_cross_entropy(y_pred, y_train)
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
    ndim_1 =  (input_neurons+1) * (hidden_neurons) + (hidden_neurons + 1) * output_neurons
    nlive = 1000 #ndim_1*100
    logl_args = [input_neurons, hidden_neurons, output_neurons]
    
    print(" ")
    print("\n NUmber of hidden neurons :::::", hidden_neurons, ":::::::::::::::::::\n")
    
    sampler = dynesty.NestedSampler(log_likelihood,
                                   prior_ptform,
                                   ndim = ndim_1,
                                   nlive = nlive,
                                   logl_args = logl_args,
                                   nparam = ndim_1,
                                   bound ='multi',
                                   sample ='hslice'
                                   ) 
    sampler.run_nested(dlogz = 8000)
    res = sampler.results
    results_list.append(res)

end = datetime.now()

print("\n Time :::", end - start)

'''Plots '''
# train metrics
x = x_train
y = y_train

logZ, accuracy_list, predictions_list = u.single_class_return_results(x, y, results_list, 
                                        x_axis, input_neurons, output_neurons)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Taiwan Dataset', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Log evidence')
plt.plot(x_axis, logZ, '-o')
plt.show()
fig.savefig('results/taiwan_log_evidence.png')

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Taiwan Dataset -Train', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Accuarcy')
plt.plot(x_axis, accuracy_list, '-o')
plt.show()
fig.savefig('results/taiwan_accuracy_train.png')
# test data set
x = x_test
y = y_test
logZ, accuracy_list, predictions_list = u.single_class_return_results(x, y, results_list, 
                                        x_axis, input_neurons, output_neurons)

plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Taiwan Dataset - Test', fontsize=16)
ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Accuracy')
plt.plot(x_axis, accuracy_list, '-o')
plt.show()
fig.savefig('results/taiwan_accuracy_test.png')
