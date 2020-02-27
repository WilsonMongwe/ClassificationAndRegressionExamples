# Nested Sampling for SLP search
Can Bayesian evidence be used for machine learning model hyper-parameter optimization?

In this short project we search for the optimal number of hidden units in a single hidden layer perceptron (SLP) using the 
Nested Sampling approach of Skilling. We document the results for the following datasets:

	0) Regression problems - Boston house price dataset 
	1) Classification problems - Iris dataset, Taiwan credit default dataset 
	
## Nested sampling

Nested sampling is an approach for calculating bayesian evidences while simultaneously computing the posterior probability distribution of the parameters. The basics steps of this algorithm are:

	0) Draw N live points from the prior distribution, and calculate their likelihoods
	1) Delete the lowest likelihood live point, and replace it with a new point drawn from the prior, but with higher likelihood
	2) Repeat 1) until some stopping criteria is met. 

## Prior

We assumed a standard normal disrtibution as the prior distribution over the parameters/weights (including the biases) for both problem types (regression and classification). This assumption can be relaxed.

## Likelihood

We set out the log likelihood functions as follows:

	a) Regression problems - the log likelihood was set to equal the negative of the mean squared error (MSE). This has the effect of us implicitly assuming that the output variable has a normal distribution.
	b) Classification problems - the log likelihood was set to equal the negative of the cross entropy loss. This has the effect of us implicitly assuming that the output variable has a multinomial distribution
	
## Neural network considerations
We assumed a tanh activation for the hidden layer for both types of problems. Classification problems had the softmax (or sigmoid) activation at the output layer. No activation function was applied at the output layer for the regression problems.

## Results
There is a high correlation between the evidence of the model (calculated using the training dataset) and the models performance on a test dataset.
