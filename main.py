import numpy as np

# Data
X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.array([[1], [1], [1], [0]])

# Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(o):
    return o * (1 - o)

# Other variables
np.random.seed(0)
weights = np.random.randn(2, 1)
bias = np.random.randn(1)

# Training loop
lr = 0.1 # lr is learning rate
for epoch in range(10000):
# Getting output
    z = np.dot(X, weights) + bias # Output
    output = sigmoid(z) # Output squashed between 0 and 1

# Getting error
    error = y - output # Error
    d_output = error * sigmoid_deriv(output) # Output ran through sigmoid_deriv


# Updating weights and bias
    weights += np.dot(X.T, d_output) * lr # Times by learning rate
    bias += np.sum(d_output) * lr

print("Predictions after training:")
print(output.round(3)) # Printing predictions after training finishes
