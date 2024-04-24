import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Input
X = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0]])

# Output
y = np.array([[1, 0, 0, 1]]).T

# Seed for random number distribution
np.random.seed(1)

# Weights initialization
synapse0 = 2 * np.random.random((3, 1)) - 1

for i in range(1000):
    # Forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, synapse0))

    # Error
    layer1_error = y - layer1

    # Multiply error by the slope of the sigmoid at the values in layer1
    layer1_delta = layer1_error * sigmoid(layer1, True)

    # Update weights
    synapse0 += np.dot(layer0.T, layer1_delta)

print("Output after training:")
print(layer1)
print("Actual output:")
print(y)

plt.plot(y, layer1, 'ro')
plt.xlabel('Actual Output')
plt.ylabel('Predicted Output')
plt.title('Backpropagation')
plt.show()
