import numpy as np
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt

data = np.loadtxt("profits-population-data.txt", delimiter=",")
plt.scatter(data[:, 0], data[:, 1])
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
#plt.show()

X = data[:, 0]
y = data[:, 1]
ones = np.ones(shape=(X.size))
X = np.c_[X,ones]
Y = np.ones(shape=(5, 2), dtype=np.int)
theta = np.zeros(shape=(2, 1)) # initialize fitting parameters
iterations = 1500;
alpha = 0.01;
#predictions = X.dot(theta)
#print(predictions)
#print(ones)

def computeCost(X, y, theta):

    # J = (1.0/(2*y.size))*(sum((X.dot(theta).flatten() - y)**2)) the entire formula

    m = y.size #number of training examples

    predictions = X.dot(theta).flatten() # multiply X matrix with theta and then flatten it to one dimension, this is the hypothesys h

    sqErrors = (predictions - y) ** 2 # give back the error squared

    J = (1.0 / (2 * m)) * sqErrors.sum() # the cost function

    return J

print(computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, num_iters):

    m = y.size #number of training examples

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta).flatten()

        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()

        J_history[i, 0] = computeCost(X, y, theta)

    return theta, J_history


X = data[:, 0]
y = data[:, 1]


#number of training samples
m = y.size  # 97 in our dataset, the number of rows

# y intercept, the constant, the value at which the fitted line crosses the y axis.
#Add a column of ones to X (interception data)
it = np.ones(shape=(m, 2))
it[:, 1] = X

#Initialize theta parameters
theta = np.zeros(shape=(2, 1))

#Some gradient descent settings
iterations = 1500
alpha = 0.01

print(compute_cost(it, y, theta))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print(theta)
#Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta).flatten()
print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
predict2 = np.array([1, 7.0]).dot(theta).flatten()
print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

#Plot the results
result = it.dot(theta).flatten()
plt.plot(data[:, 0], result)
#plt.show()
