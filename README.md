# Nested Sampling for SLP search
Can Bayesian evidence be used for machine learning models selection?

In this mini project we search for the optimal number of hidden units in a single hidden layer perceptron (SLP) using the 
Nested Sampling approach of Skilling. We document the results for the following datasets:

	0) Boston house price dataset - which is a regression problem
	1) Iris dataset - which is a classification problem

Prior

We assumed that a standard normal disrtibution as the prior distribution for both datasets.

Likelihood

We set out the log likelihood functions as follows:

	0) Boston house price dataset - the log likelihood was set to equal the negative of the mean squared error (MSE). This has the effect of us implicitly assuming 
	that the output variable has a normal distribution
	1) Iris dataset - the log likelihood was set to equal the negative of the cross entropy loss. This has the effect of us implicitly assuming that
	the output variable has a multinomial distribution
