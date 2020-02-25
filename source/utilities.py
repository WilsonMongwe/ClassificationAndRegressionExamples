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


def categorical_accuracy(y_true, y_pred):
    correct = 0
    total = 0
    for i in range(len(y_true)):
        act_label = np.argmax(y_true[i]) # act_label = 1 (index)
        pred_label = np.argmax(y_pred[i]) # pred_label = 1 (index)
        if(act_label == pred_label):
            correct += 1
        total += 1
    accuracy = (correct/total)
    return accuracy

def sigmoid(x):
    return scipy.special.expit(x)
   
def relu(x):
    return x * (x > 0)

def softmax(x):
    #return scipy.special.softmax(x, axis =1)
    e_x = np.exp(x - np.max(x))
    return (e_x.T / e_x.sum(axis=1)).T 

    
def multi_cross_entropy(predictions, targets):   
    #N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9)) #/N
    return ce
    
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = sklearn.metrics.log_loss(targets, predictions)
    return ce * N

def mean_squared_error(predictions, targets):
    N = predictions.shape[0]
    mse = metrics.mean_squared_error(predictions, targets)
    return mse * N

def neural_network(X, w_1, w_2):
    hidden_layer = np.tanh(X @ w_1)
    y = sigmoid(hidden_layer @ w_2)
    return y

def regression_neural_network(X, w_1, w_2):
    hidden_layer = np.tanh(X @ w_1)
    y = hidden_layer @ w_2
    return y

def predictions(X, W, neurons_1, neurons_2, output_neurons):
    param_1 = neurons_1 * neurons_2
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((neurons_1,neurons_2))
    
    param_2 = neurons_2 * output_neurons
    w_2 = W[param_1:param_1 + param_2]
    w_2 = w_2.reshape((neurons_2,output_neurons))

    predictions = neural_network(X, w_1, w_2)
    
    return predictions


def multi_neural_network(X, w_1, b_1, w_2, b_2):
    hidden_layer = np.tanh((X @ w_1) + b_1.T)
    y = softmax((hidden_layer @ w_2) + b_2.T)
    return y


def multi_predictions(X, W, neurons_1, neurons_2, output_neurons): 
    param_1 = neurons_1 * neurons_2
    bias_1  =  neurons_2
    w_1 = W[0:param_1]
    w_1 = w_1.reshape((neurons_1,neurons_2))
    b_1 = W[param_1:param_1+ bias_1]
    
    param_2 = neurons_2 * output_neurons
    bias_2 = output_neurons
    w_2 = W[param_1 + bias_1:param_1 + bias_1 + param_2]
    w_2 = w_2.reshape((neurons_2, output_neurons))
    b_2 = W[param_1 + bias_1 + param_2:param_1+ 
            bias_1 + param_2 + bias_2]

    predictions = multi_neural_network(X, w_1, b_1, w_2, b_2)
    
    return predictions


def return_results_regression(x, y, results_list, x_axis, neurons_1, 
                                        neurons_2, output_neurons):
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
        neurons_2 = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        W, cov = dynesty.utils.mean_and_cov(samples, weights)
        #W = samples[index_max_list[i-1]]
        predictions_list.append(predictions(x, W, neurons_1, 
                                        neurons_2, output_neurons))
    
    accuracy_list = []
    
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        accuracy_list.append(metrics.mean_squared_error(y, y_pred))
        
    return logZ, accuracy_list


def return_results(x, y, results_list, x_axis, neurons_1, 
                                        neurons_2, output_neurons):
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
        neurons_2 = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        W, cov = dynesty.utils.mean_and_cov(samples, weights)
        #W = samples[index_max_list[i-1]]
        predictions_list.append(np.round(predictions(x, W, neurons_1, 
                                        neurons_2, output_neurons)))
    
    accuracy_list = []
    f1_list = []
    roc_auc_list = []
    
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        accuracy_list.append(metrics.accuracy_score(y, y_pred))
        roc_auc_list.append(metrics.roc_auc_score(y, y_pred))
        f1_list.append(metrics.f1_score(y, y_pred))
        
    return logZ, accuracy_list, f1_list, roc_auc_list



def multi_return_results(x, y, results_list, x_axis, neurons_1, 
                                        neurons_2, output_neurons):
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
        neurons_2 = i
        samples = samples_list[i-1]
        weights = weights_list[i-1]
        #W, cov = dynesty.utils.mean_and_cov(samples, weights)
        W = samples[index_max_list[i-1]]
        predictions_list.append(predictions(x, W, neurons_1, 
                                        neurons_2, output_neurons))
    
    accuracy_list = []
    f1_list = []
    roc_auc_list = []

    
    for i  in x_axis:
        y_pred = predictions_list[i-1]
        accuracy_list.append(categorical_accuracy(y, y_pred))
        #roc_auc_list.append(metrics.roc_auc_score(y, y_pred))
        #f1_list.append(metrics.f1_score(y, y_pred))
        
    return logZ, accuracy_list, f1_list, roc_auc_list


''' Loading in the data'''

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	#minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def convert_to_int(targets):
     ls =[]
     for x in targets:
         ls.append(int(x))
     return ls
 
def one_hot_encoding(targets):
    integer_encoded= convert_to_int(targets.tolist())
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(3)]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def return_seeds_processed_data(filename):
    # load and prepare data
    #filename = "data.csv"
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
    	str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    
    data = np.array(dataset)
    X = data[:,0:6] 
    y = np.array(one_hot_encoding(data[:,7]))
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_taiwan_processed_data():
    #Load the data
    dataset = pd.read_excel("default_data.xls")
    dataset.index = dataset['ID']
    dataset.drop('ID',axis=1,inplace=True)
    dataset['SEX'].value_counts(dropna=False)
    dataset['EDUCATION'].value_counts(dropna=False)
    dataset = dataset.rename(columns={'PAY_0': 'PAY_1'})
    
    # Clean the data
    fil = (dataset.EDUCATION == 5) | (dataset.EDUCATION == 6) | (dataset.EDUCATION == 0)
    dataset.loc[fil, 'EDUCATION'] = 4
    dataset['EDUCATION'].value_counts(dropna = False)
    dataset.loc[dataset.MARRIAGE == 0, 'MARRIAGE'] = 3
    
    fil = (dataset.PAY_1 == -1) | (dataset.PAY_1==-2)
    dataset.loc[fil,'PAY_1']=0
    dataset.PAY_1.value_counts()
    fil = (dataset.PAY_2 == -1) | (dataset.PAY_2==-2)
    dataset.loc[fil,'PAY_2']=0
    dataset.PAY_2.value_counts()
    fil = (dataset.PAY_3 == -1) | (dataset.PAY_3==-2)
    dataset.loc[fil,'PAY_3']=0
    dataset.PAY_3.value_counts()
    fil = (dataset.PAY_4 == -1) | (dataset.PAY_4==-2)
    dataset.loc[fil,'PAY_4']=0
    dataset.PAY_4.value_counts()
    fil = (dataset.PAY_5 == -1) | (dataset.PAY_5==-2)
    dataset.loc[fil,'PAY_5']=0
    dataset.PAY_5.value_counts()
    fil = (dataset.PAY_6 == -1) | (dataset.PAY_6==-2)
    dataset.loc[fil,'PAY_6']=0
    dataset.columns = dataset.columns.map(str.lower)
     
    #Standardize the numerical columns
    col_to_norm = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4',
           'bill_amt5', 'bill_amt6', 'pay_amt1', 'pay_amt2', 'pay_amt3',
           'pay_amt4', 'pay_amt5', 'pay_amt6']
    dataset[col_to_norm] = dataset[col_to_norm].apply(lambda x : (x-np.mean(x))/np.std(x))
    
    #Split the data into training and test set
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1)
    
    return x_train, y_train, x_test, y_test

def return_iris_processed_data():
      # Load digits dataset
    iris = datasets.load_iris()
    
    X = iris['data']
    y = iris['target']
    
    # One hot encoding
    enc = OneHotEncoder()
    Y = enc.fit_transform(y[:, np.newaxis]).toarray()
    
    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state =1)
    
    return x_train, y_train, x_test, y_test

def return_boston_processed_data():
      # Load digits dataset
    X, Y = datasets.load_boston(return_X_y=True)
        
      
    # Scale data to have mean 0 and variance 1 
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data set into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size = 0.3, random_state =1)
    
    return x_train, y_train, x_test, y_test

def return_mnist_processed_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    x_train = x_train.reshape(60000, 784)
    x_test =  x_test.reshape(10000, 784)
    x_train =x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    y_train = np.reshape(y_train, (60000,num_classes))
    y_test = np.reshape(y_test, (10000,num_classes))
    
    return x_train, y_train, x_test, y_test