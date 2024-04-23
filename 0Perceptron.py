class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, threshold=0):
        self.weights = [0] * num_inputs
        self.learning_rate = learning_rate
        self.threshold = threshold

    def predict(self, inputs):
        activation = sum(weight * input_val for weight, input_val in zip(self.weights, inputs))
        return 1 if activation > self.threshold else 0

    def train(self, training_inputs, labels, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]


# Example usage:

# Define inputs, weights, and labels for the AND gate
and_training_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
and_labels = [0, 0, 0, 1]

# Create a Perceptron instance for AND gate
and_perceptron = Perceptron(num_inputs=2)

# Train the Perceptron for AND gate
and_perceptron.train(and_training_inputs, and_labels)

# Test the trained AND gate Perceptron
print("AND Gate Perceptron:")
print(and_perceptron.predict([0, 0]))  # 0
print(and_perceptron.predict([0, 1]))  # 0
print(and_perceptron.predict([1, 0]))  # 0
print(and_perceptron.predict([1, 1]))  # 1

# Define inputs, weights, and labels for the OR gate
or_training_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
or_labels = [0, 1, 1, 1]

# Create a Perceptron instance for OR gate
or_perceptron = Perceptron(num_inputs=2)

# Train the Perceptron for OR gate
or_perceptron.train(or_training_inputs, or_labels)

# Test the trained OR gate Perceptron
print("\nOR Gate Perceptron:")
print(or_perceptron.predict([0, 0]))  # 0
print(or_perceptron.predict([0, 1]))  # 1
print(or_perceptron.predict([1, 0]))  # 1
print(or_perceptron.predict([1, 1]))  # 1
