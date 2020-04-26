import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import random
import utilities as u
import pandas as pd
# the data, split between train and test sets
random.seed(1)
x_train, y_train, x_test, y_test = u.return_boston_processed_data()


# Read in alpha and beta from GA results
alpha = pd.read_excel("alpha_boston_mean.xlsx").values
beta = pd.read_excel("beta_boston_mean.xlsx").values
alpha = alpha.reshape(30, 1)
beta = beta.reshape(30, 1)

kf = KFold(n_splits = 10)
scores = []
x_axis = range(1, 31, 1) # 


for i in x_axis:
    scored_temp =[]
    a = alpha[i-1][0]
    clf = MLPRegressor(activation = 'tanh', alpha = a, hidden_layer_sizes=(i), 
                   random_state=1, max_iter=100000)
    
    print("\n Number of hidden neurons :::::", i, ":::::::::::::::::::\n")
    for train_indices, test_indices in kf.split(x_train):
        clf.fit(x_train[train_indices], y_train[train_indices])
        y_pred = clf.predict(x_train[test_indices])
        scored_temp.append(metrics.mean_squared_error(y_pred,y_train[test_indices]))
    scores.append(np.array(scored_temp).mean())
    print("Score", scores[i-1])


import matplotlib.pyplot as plt
plt.style.use(['bmh'])
fig, ax = plt.subplots(1)
fig.suptitle('Train Dataset', fontsize=16)
ax.set_xlabel('Number of hidden neurons')
ax.set_ylabel('Average MSE over 10-Folds')
plt.plot(x_axis, scores, '-o', label="MSE")
plt.legend(loc = 4)
plt.show()

print("Score", scores)