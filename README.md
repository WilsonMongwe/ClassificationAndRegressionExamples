# Nested Sampling for SLP search
Can Bayesian evidence be used for machine learning model selection?

In this mini project we search for the optimal number of hidden units in a single hidden layer perceptron (SLP) using the 
Nested Sampling approach of Skilling. We document the results for the following datasets:

	0) Boston house price dataset - which is a regression problem
	1) Iris dataset - which is a classification problem

## Prior

We assumed a standard normal disrtibution as the prior distribution over the weights (including the biases) for both problem types (regression and classification).

## Likelihood

We set out the log likelihood functions as follows:

	a) Regression problems - the log likelihood was set to equal the negative of the mean squared error (MSE). This has the effect of us implicitly assuming that the output variable has a normal distribution.
	b) Classification problems - the log likelihood was set to equal the negative of the cross entropy loss. This has the effect of us implicitly assuming that the output variable has a multinomial distribution
	
## Neural network comments
We assumed a tanh activation for the hidden layer for both types of problems. Classification problems had the softmax (or sigmoid) activation at the output layer. No activation function was applied at the output layer for the regression problems.
