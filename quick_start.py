import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.linear_model as lm
import sklearn.model_selection as ms
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


# Let's write a function to look at the residuals
def visualize_residuals(model, x, y):
    ypred = model.predict(x)
    plt.plot(y-ypred, 'o', label='Errors')
    plt.plot(np.arange(len(y)), np.zeros(len(y)), '-', label='Ideal')
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title('R2: ' + str(model.score(x, y)))
    plt.show()

def split(X, y, test_size=0.5):
	X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=test_size, random_state=42)
	return X_train, X_test, y_train, y_test


def pca(X_train):
	pass

def preprocessing(X_train, X_test, standardize=False):
	# put X_test and y_test in a "box" for later.
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_train)

	return X_train, X_test


def ridge(X_train, y_train, alphas, kf, plot_alphas=False):
	# use k-fold validation on each value of alpha to determine the mean R^2.
	ridge_scores = []

	for alpha in alphas:
	    # initialize a ridge object below with the current alpha
	    this_alpha_scores = []
	    # iterate the n folds for cross validation
	    for train, validate in kf.split(X_train):
	        ridge = lm.Ridge(alpha=alpha)
	        # fit the ridge object on the training set and score on the validation set 
	        scores = ridge.fit(X_train[train], y_train[train]).score(X_train[validate], y_train[validate])
	    
	        this_alpha_scores.append(scores)
	    ridge_scores.append(this_alpha_scores)
	    
	ridge_scores = np.vstack(ridge_scores)

	ridge_bestalpha = alphas[ridge_scores.mean(1) == ridge_scores.mean(1).max()]  # the best alpha is the one the produces the highest score

	if plot_alphas == True:
				# plot the mean score against alpha candidates
		plt.figure(figsize=(6, 3))
		plt.plot(alphas, ridge_scores.mean(1), label='scores')
		plt.plot(ridge_bestalpha, ridge_scores.mean(1).max(), 'ro', label='alpha: ' + str(ridge_bestalpha))
		plt.xscale('log')
		plt.xlabel('alpha')
		plt.ylabel('k-fold R-squared')
		plt.title('Optimizing Ridge')
		plt.legend()
		plt.show()

	return ridge_scores, ridge_scores.mean(1).max(), ridge_bestalpha


def lasso(X_train, y_train, alphas, kf, plot_alphas=False):
	# use k-fold validation on each value of alpha to determine the mean R^2.
	lasso_scores = []

	for alpha in alphas:
	    this_alpha_scores = []
	    for train, validate in kf.split(X_train):
	        # initialize a ridge object below with the current alpha
	        lasso = lm.Lasso(alpha=alpha)
	        # fit the ridge object on the training set and score on the validation set 
	        scores = lasso.fit(X_train[train], y_train[train]).score(X_train[validate], y_train[validate]) 
	    
	        this_alpha_scores.append(scores)
	    lasso_scores.append(this_alpha_scores)
	    
	lasso_scores = np.vstack(lasso_scores)

	lasso_bestalpha = alphas[lasso_scores.mean(1) == lasso_scores.mean(1).max()]  # the best alpha is the one the produces the highest scoreridge_bestalpha = alphas[ridge_scores.mean(1) == ridge_scores.mean(1).max()]  # the best alpha is the one the produces the highest score

	if plot_alphas == True:
		# plot the mean score against alpha candidates
		plt.figure(figsize=(6, 3))
		plt.plot(alphas, lasso_scores.mean(1), label='scores')
		plt.plot(lasso_bestalpha, lasso_scores.mean(1).max(), 'ro', label='alpha: ' + str(ridge_bestalpha))
		plt.xscale('log')
		plt.xlabel('alpha')
		plt.ylabel('k-fold R-squared')
		plt.title('Optimizing LASSO')
		plt.legend()
		plt.show()


	return lasso_scores, lasso_scores.mean(1).max(), lasso_bestalpha





