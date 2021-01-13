class FixedTopologyNeuralNetwork:

    class Neuron:
        def __init__(self, wages, bias):
            self.wages = wages
            self.bias = bias


    #*    
    #*   layers = list of pairs (number of neurons in layer, activator functions)
    #*   wages = list/array of floats, concatenation of wages and biases for each neuron in layers
    #*  
    def __init__(self, input_size, layers, wages):
        self.input_size = input_size
        self.layers = []

        prev_layer_size = input_size
        id = 0

        for layer_size, func in layers:
            neurons = []

            for neuron_id in range(layer_size):
                wages = np.array(wages[id:id + prev_layer_size])
                bias = wages[id + prev_layer_size]
                
                id += prev_layer_size + 1

                neurons.append(self.Neuron(wages, bias))


            self.layers.append((neurons, func))
            prev_layer_size = layer_size


    def eval(input):
        output = np.array(input)

        for neurons, func in self.layers:
            output = func(np.array([np.sum(neuron.wages * output) + bias for neuron in neurons]))
        
        return output
