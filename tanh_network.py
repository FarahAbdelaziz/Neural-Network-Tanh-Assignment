import numpy as np
def tanh(x):
    return np.tanh(x)
input_data = np.array([0.05, 0.10])
hidden_layer_weight = np.random.uniform(-0.5, 0.5, (2, 2))
output_layer_weight = np.random.uniform(-0.5, 0.5, (2, 2))

b1 = 0.5
b2 = 0.7

hidden_net = np.dot(hidden_layer_weight, input_data) + b1
hidden_output = tanh(hidden_net)

output_net = np.dot(output_layer_weight, hidden_output) + b2
final_output = tanh(output_net)

print("Input:", input_data)
print("Hidden Layer Output:", hidden_output)
print("Final Output:", final_output)



