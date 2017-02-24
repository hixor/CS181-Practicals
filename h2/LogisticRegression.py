import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
import pandas as pd
from sklearn import preprocessing

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter=0):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.theta = None
        self.deltas = None
        self.probs = None
        self.grad = None
    
    def softmax(self,mat,k):
        exps = np.exp(mat)
        denom = np.array([np.sum(exps,1), ]*k).transpose()
        return exps/denom


    def fit(self, X, C):
        X = np.column_stack((np.ones(len(X)), X))
        X = preprocessing.scale(X)
        
        n = X.shape[0]
        m = X.shape[1]
        k = 3
        num_iters = 1000
        theta = np.zeros((m,k))

        
        for i in range(num_iters):
            h = np.dot(X,theta) 
            probs = self.softmax(h,k)
            deltas = probs - np.array(pd.get_dummies(C))
            grad = np.dot(X.T, deltas)/len(X) + (self.lambda_parameter/m)*theta
            if i == 0:
                self.deltas = deltas
                self.grad = grad
            theta = theta - self.eta*grad
        self.probs = probs
        self.theta = theta
        
        return self

    def predict(self, X_to_predict):
        X_to_predict = np.column_stack((np.ones(len(X_to_predict)), X_to_predict))
        h_pred = np.dot(X_to_predict, self.theta)
        pred_probs = self.softmax(h_pred,3)
        return np.argmax(pred_probs,axis=1)

    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))
        
        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
