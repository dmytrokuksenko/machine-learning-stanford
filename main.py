"""
    Andrew Ng Machine Learning Online Class - Exercise 1: Linear Regression.

    @author Dmytro Kuksenko
    @date March 26, 2021
"""
import numpy as np
import matplotlib.pyplot as plt

def plotData(x, y):
    """
    Plot x vs. y data using Matplotlib
    :param x:
    :param y:
    """
    plt.plot(x, y, 'rx') # Plot the data
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter plot of training data')
    plt.show()

def computeCost(x, y, theta):
    """
    :param x:
    :param y:
    :param theta:
    :return:
    """
    m = np.prod(y.shape)
    h = np.zeros(m)
    J = 0

    for i in range(m):
        h[i] = theta[0]*x[i][0] + theta[1]*x[i][1]

    J = np.sum((h-y)**2)/(2*m)


    return J

# Dataset for linear regression analysis
data = np.loadtxt("ex1data1.txt", delimiter=',')
# Population of the city
x = data[:, 0]
# Profit of a food truck
y = data[:, 1]

plotData(x, y)
m = np.prod(x.shape)
x = np.vstack((np.ones((m)), x)).T
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

J = computeCost(x, y, theta)
print("The first cost is: %.2f" %J)

theta = np.array([-1, 2])
J = computeCost(x, y, theta)
print("The first cost is: %.2f" %J)




