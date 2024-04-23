class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights")

        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        return 1 if weighted_sum > 0 else 0  # Change to > 0 instead of >= threshold


class McCullochPittsLogicGate:
    def __init__(self, weights):
        self.neuron = McCullochPittsNeuron(weights, 0)  # Threshold set to 0 for simplicity

    def evaluate(self, inputs):
        return self.neuron.activate(inputs)


# Example usage:

input_vector = [0, 1, 1]
weight_vector = [-1, 1, 1]

neuron = McCullochPittsNeuron(weights=weight_vector, threshold=0)
print("Input vector:", input_vector)
print("Weight vector:", weight_vector)
print("Dot product:", sum(w * x for w, x in zip(weight_vector, input_vector)))
print("Activation:", neuron.activate(input_vector))
