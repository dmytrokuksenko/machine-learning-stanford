"""
    Andrew Ng Machine Learning Online Class - Exercise 1: Linear Regression.

    @author Dmytro Kuksenko
    @date March 26, 2021
"""
import numpy as np
import matplotlib.pyplot as plt

def plotData(x, y):
    """
    Plot x vs. y data points using Matplotlib
    """
    plt.plot(x, y, 'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter plot of training data')
    plt.show()

def computeCost(x, y, theta):
    """
    Computes the cost functions

    :param x: vector with features
    :param y: target variable
    :param theta: fitting parameters
    :return: estimation of the cost function
    """
    n = np.prod(y.shape) # Number of training examples

    h = np.zeros(n) # Initialization hypothesis vector with zeros

    for i in range(n):
        h[i] = theta[0]*x[i][0] + theta[1]*x[i][1]

    J = np.sum((h-y)**2)/(2*n)

    return J

def gradientDecent(x, y, theta, alpha, iterations):
    """
    Computes the cost function using Gradient Decent

    :param x: n-dimensional vector with features
    :param y: vector with target variable
    :param alpha: learning rate
    :param theta: fitting coefficients
    :param iterations: total number of iterations for Gradient Decent
    :return: cost function
    """

    n = np.prod(y.shape) # Number of training examples
    J_history = np.zeros(iterations)
    h = np.zeros(n)  # Initialization hypothesis vector with zeros

    for i in range(iterations):

        for m in range(n):
            h[m] = theta[0] * x[m][0] + theta[1] * x[m][1]

        theta[0] = theta[0] - alpha * (np.sum(np.multiply((h - y), x[:, 0])) / n)
        theta[1] = theta[1] - alpha * (np.sum(np.multiply((h - y), x[:, 1])) / n)

        J_history[i] = computeCost(x, y, theta)

    return theta


# Dataset for linear regression analysis
data = np.loadtxt("ex1data1.txt", delimiter=',')
# Population of the city
x = data[:, 0]
# Profit of a food truck
y = data[:, 1]

# plotData(x, y)
m = np.prod(x.shape)
x = np.vstack((np.ones(m), x)).T
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

# Compute initial cost with fitting parameters all zero
J = computeCost(x, y, theta)
J_anl = 32.07 # cost function ca
print("The expected vs. calculated cost is: %.2f vs. %.2f" %(J, J_anl))

# Compute initial cost with non-zero fitting parameters
theta = np.array([-1, 2])
J_anl = 54.24
J = computeCost(x, y, theta)
print("The expected vs. calculated cost is: %.2f vs. %.2f" %(J, J_anl))

# Compute fitting parameters using gradient decent
theta = np.zeros((2,1))
theta = gradientDecent(x, y, [0, 0], alpha, iterations)

print("Fitting parameters computed from Gradient Decent are: %.2f and %.2f" %(theta[0], theta[1]))

# Plot the predicted results
plt.plot(x, y, 'rx')
plt.plot(x[:, 1])
plt.show()

