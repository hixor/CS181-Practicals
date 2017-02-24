from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import pandas as pd

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    
    def __fruit_data(self):
        self.fruit = pd.read_csv("fruit.csv")
        self.X = self.fruit[["width","height"]]
        self.y = self.fruit["fruit"]

    
     def fit(self, X, Y):
        self.X = pd.DataFrame(X)
        self.Y = pd.DataFrame(Y)
        c1 = self.X[self.y==1]
        c2 = self.X[self.y==2]
        c3 = self.X[self.y==3]
        self.mu1 = np.mean(c1)
        self.mu2 = np.mean(c2)
        self.mu3 = np.mean(c3)
        if self.isSharedCovariance:
            self.sigma1 = np.cov(X.transpose())
            self.sigma2 = np.cov(X.transpose())
            self.sigma3 = np.cov(X.transpose())
        else:
            self.sigma1 = np.cov(c1.transpose())
            self.sigma2 = np.cov(c2.transpose())
            self.sigma3 = np.cov(c3.transpose())
        return (self.sigma, self.mu1, self.mu2, self.mu3)

 

        def predict(self, X_to_predict):

        p1 =[]; p2=[]; p3=[]
        pred_x = pd.DataFrame(X_to_predict)
        for x in pred_x.iterrows():
            p1.append(multivariate_normal.pdf(x[1], mean=self.mu1, cov=self.sigma1))
            p2.append(multivariate_normal.pdf(x[1], mean=self.mu2, cov=self.sigma2))
            p3.append(multivariate_normal.pdf(x[1], mean=self.mu3, cov=self.sigma3))
        preds = pd.DataFrame({"Class 1":p1, "Class 2": p2, "Class 3": p3})
               
        return np.argmax(np.array(preds),axis=1)    


    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .005), np.arange(y_min,
            y_max, .005))

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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
