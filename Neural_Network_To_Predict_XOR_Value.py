import numpy as np




class XOR_Predictor:

    def __init__(self, network_layers, activation_function_selected='tanh'):
        if activation_function_selected == 'sigmoid':
            self.activation_function_selected = sigmoid
            self.activation_function_selected_prime = sigmoid_prime
        elif activation_function_selected == 'tanh':
            self.activation_function_selected = tanh
            self.activation_function_selected_prime = tanh_prime

        print("Layers : ",len(network_layers)-1)
        # print("Layers : ",(network_layers[1-1] + 1, network_layers[1] + 1))

        # Initialization of Weights
        self.weights = []
        # network_layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden network_layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(network_layers) - 1):
            print("i : ",i)
            r = np.random.random((network_layers[i-1] + 1, network_layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = np.random.random( (network_layers[i] + 1, network_layers[i+1])) - 1
        self.weights.append(r)


        print("Weights : ",self.weights)


    def fit(self, Network_Inputs, Expected_Output, epochs=100000, learning_rate=0.2):
        Network_Inputs = np.concatenate((bias_column.T, Network_Inputs), axis=1)
        bias_column = np.atleast_2d(np.bias_column(Network_Inputs.shape[0]))
         
        for k in range(epochs):
            i = np.random.randint(Network_Inputs.shape[0])
            a = [Network_Inputs[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation_function_selected = self.activation_function_selected(dot_value)
                    a.append(activation_function_selected)
            # output layer
            error = Expected_Output[i] - a[-1]
            deltas = [error * self.activation_function_selected_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_function_selected_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation_function_selected 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)


    def predict_output(self, x): 
        a = np.concatenate((np.bias_column(1).T, np.array(x)), axis=0)      
        for l in range(0, len(self.weights)):
            a = self.activation_function_selected(np.dot(a, self.weights[l]))
        return a



# Tanh Activation
def tanh_prime(x):
    return 1.0 - x**2

def tanh(x):
    return np.tanh(x)


# Sigmoid Activation
def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))



if __name__ == '__main__':

    nn = XOR_Predictor([2,2,1])
    Network_Inputs = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Expected_Output = np.array([0, 1, 1, 0])
    nn.fit(Network_Inputs, Expected_Output)
    for e in Network_Inputs:
        print(e,nn.predict_output(e))