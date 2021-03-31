"""
Curve fitting functions: Gaussian Process and Linear Regression
"""

import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import sklearn.gaussian_process.kernels as K
from sklearn.linear_model import LinearRegression

def GP(x, y, x_eval):
    kernel = C(1.0, (1e-2, 1e2)) * RBF(100, (1e-2, 1e2))
    #kernel = K.RationalQuadratic(length_scale=1, alpha=0.5)
    #kernel = K.Matern(length_scale=1.0, nu=2.0)
    #kernel = RBF(length_scale=100)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
    X = x.reshape(len(x), 1)
    gp.fit(X, y)
    X_eval = x_eval.reshape(len(x_eval), 1)
    y_pred, sigma = gp.predict(X_eval, return_std=True)
    score = gp.score(X, y)
    return y_pred, sigma, score

def LR(x, y, x_eval):
    X = x.reshape(len(x), 1)
    reg = LinearRegression().fit(X, y)
    score = reg.score(X, y)
    y_pred = reg.predict(x_eval.reshape(len(x_eval), 1))
    a = reg.coef_[0]
    b = reg.intercept_
    a_up = 0.00
    a_up_nointer = 0.00 

    while True:
        y_up = (a+a_up)*x + b
        if sum(y_up >= y) >= 0.9*len(y):
            break
        a_up += 0.01

    while True:
        y_up = (a+a_up_nointer)*x
        if sum(y_up >= y) >= 0.9*len(y):
            break
        a_up_nointer += 0.01

    label_nom = 'y = '+str(np.round(a,2))+'x + '+str(np.round(b,2))
    y_pred_up = (a+a_up)*x_eval + b
    y_pred_up_nointer = (a+a_up_nointer)*x_eval
    label_up = 'y = '+str(np.round(a+a_up,2))+'x + '+str(np.round(b,2))
    label_up_nointer = 'y = '+str(np.round(a+a_up_nointer,2))+'x'

    return y_pred, score, label_nom, y_pred_up, label_up, y_pred_up_nointer, label_up_nointer

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # x = np.array([1, 2, 3, 4, 5])
    # y = 0.68*x + 0.20 + 0.40*np.random.normal(size=5)
    # x_eval = np.linspace(1, 5, 100)
    
    x = np.load('./output/hope_result/x2_199.npy')
    y = np.load('./output/hope_result/y2_199.npy')
    index = x <= 0.30
    x = x[index]
    y = y[index]
    x_eval = np.linspace(0, np.max(x), 200)

    #y_pred, sigma, score = GP(x, y, x_eval)

    y_pred, score, label_nom, y_pred_up, label_up, y_pred_up_nointer, label_up_nointer = LR(x, y, x_eval)

    plt.figure()
    plt.plot(x, y, 'b.', markersize=5, label='data')
    plt.plot(x_eval, y_pred, 'b-', label=label_nom+' (nominal)')
    plt.plot(x_eval, y_pred_up, 'r', linestyle='--', label=label_up+' (cover 90% data)')
    plt.plot(x_eval, y_pred_up_nointer, 'r', linestyle='dotted', label=label_up_nointer+' (cover 90% data)')
    plt.xlabel('add disagreement (b, c)')
    plt.ylabel('add error (b)')
    plt.title('Parmesan')
    #plt.title(str(score))
    #plt.fill_between(x_eval, y_pred-1.96*sigma, y_pred+1.96*sigma, label='confidence interval')
    plt.legend()
    plt.show()



