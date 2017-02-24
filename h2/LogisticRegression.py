import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
import pandas as pd

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    def oneHot(self, y, k=3):
        base = np.array([np.ones(len(y)),]*k).transpose()
        scale = np.array(range(1,k+1))
        comp = base*scale
        y_comp = np.array([np.array(y).transpose(),]*k).transpose()
        self.y_mat = 1*np.equal(y_comp,comp)
        return self.y_mat
    
    def softmax(self,mat,k):
        exps = np.exp(mat)
        denom = np.array([np.sum(exps,1), ]*k).transpose()
        return exps/denom

    def fit(self, X, C):
        self.X = pd.DataFrame(X)
        self.C = pd.DataFrame(C)
        self.y = self.oneHot(self.C,3)
        
        n = self.X.shape[0]
        m = self.X.shape[1]
        k = 3
        num_iters = 100
        theta = np.zeros((m,k))
        mat = np.dot(X,theta)
        
        for i in range(num_iters):
            h = np.dot(X,theta) 
            probs = softmax(h,k)
            grad = (float(1)/m)*np.dot(X.transpose(),y_mat - probs)/len(X) + (self.lambda_parameter/m)*theta
            theta = theta - self.eta*grad
        
        return theta

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        
        h_pred = np.dot(X_to_predict,theta)
        pred_probs = softmax(h_pred,k)
        
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
