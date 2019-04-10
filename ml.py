import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.model_selection import KFold, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier


# Keras and Stuff
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam

# Keras model
class KerasBase:
    def __init__(self, input_size, num_classes = 3):
        self.num_classes = num_classes
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_shape=(input_size,)))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(4, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
    
    def fit(self, X_train, y_train, verbose=0, summary=False, batch_size = 64, epochs = 100, validation_split=0.2):
        
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
        
        if summary == True:
            self.model.summary()
        
        history = self.model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_split=validation_split)
        return self
        
    def score(self, X_test, y_test, verbose=0):
        # convert class vectors to binary class matrices

        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        score = self.model.evaluate(X_test, y_test, verbose=verbose)
        return score
    
    def predict(self, X_test):
        pred = self.model.predict_classes(X_test)
        return pred
    



def adaboost_grid_cv(X_train_features, y_train, n_estimators=[1000], 
                         learning_rate=[0.001, 0.01, 0.1, 1],
                         cv=5, n_jobs=-1, scoring='roc_auc', verbose=3):
    param_grid = [ {'n_estimators': n_estimators,
    'learning_rate': learning_rate},
    ]

    ada = GridSearchCV(AdaBoostClassifier(),
    param_grid=param_grid, cv=cv, n_jobs=n_jobs, scoring=scoring, verbose=verbose)
    ada.fit(X_train_features, y_train)

    print(ada.best_params_)
    return ada


def gradient_boosting_grid_cv(X_train_features, y_train, n_estimators=[1000], 
                         learning_rate=[0.01, 0.1, 1], max_depth=[1,2,3,4], 
                         min_samples_split=[2,3,4],
                         cv=5, n_jobs=-1, scoring='roc_auc', verbose=3):
    param_grid = [ {'n_estimators': n_estimators,
    'learning_rate': learning_rate, 'max_depth': max_depth,
    'min_samples_split': min_samples_split},
    ]

    gb = GridSearchCV(GradientBoostingClassifier(),
    param_grid=param_grid, cv=cv, n_jobs=n_jobs, scoring=scoring, verbose=verbose)
    gb.fit(X_train_features, y_train)

    print(gb.best_params_)
    return gb





def create_submission(filename, pred):
    submission = pd.read_csv('sample_submission.csv')
    submission.loc[:,'Predicted'] = pred.astype(int)
    submission.to_csv(filename, header=True, index=False)

def get_col_types(df):
    num_cols = []
    cat_cols = []
    columns = df.columns
    for col in columns:
        if df[col].dtype == 'object':
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols

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





