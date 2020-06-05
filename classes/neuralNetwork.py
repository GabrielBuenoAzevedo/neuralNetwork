import numpy as np

#This file contains the class responsible to constructing a neural network.

class NeuralNetwork:
  def __init__(self, sizes, layer_functions):
    self.sizes = sizes  #Sizes is array containing the sizes of each layer
    self.biases = [ np.random.randn(size) for size in sizes[1:] ]
    self.weights = [ np.random.randn(size, last_size) for (last_size, size) in zip(sizes, sizes[1:]) ]
    self.layer_functions = layer_functions
    self.node_values = []

    for (last_size, size) in zip(sizes, sizes[1:]):
      print(size, last_size) 

  def print(self):
    for weight in self.weights:
      print('-----')
      print(weight)
    print('~~~~~~~~~~~~~~~~~~~~')
    for weight in self.biases:
      print('--------------')
      print(weight)

  def applyFunction(self, input, functionName):
    if (functionName == 'softmax'):
      return self.softmax(input)
    return input

  def softmax(self, input):
    exp = np.exp(input)
    print(exp.sum(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

  def crossEntropy(self, result, output):
    return np.sum(result*np.log(output))

  def feedforward(self, input):
    node_values = [input]
    for (biases, weights, layer_function) in zip(self.biases, self.weights, self.layer_functions):
      input = np.dot(input, weights.T) + biases
    #   # input = weights.dot(input) + biases
      print(input)
      print('---------------')
      input = self.applyFunction(input, layer_function)
      node_values.append(input)
    self.node_values = node_values
    return input

  def crossEntropyBackpropagation(self, inputs, outputs, results):
    results = outputs - results
    cost_weights = np.dot(inputs, results)
    print(cost_weights)

  def training(self, training_set, max_epochs, training_results):
    for i in range(0, max_epochs):
      weights_cost = []
      output = self.feedforward(training_set)
      weights = self.weights  
      node_values = self.node_values
      error = self.crossEntropy(training_results, output)

      for i in range(len(weights)-1, -1, -1):
        print('----------------------')
        self.crossEntropyBackpropagation(node_values[i], )
        print(node_values[i+1])
        print(weights[i])
        print(node_values[i])
        print('----------------------')

      # for layer in reversed(self.node_values):
      #   print(layer)
      #   print('aaaaa')
      # print('-----')
      # print(len(self.node_values))
      # print(len(self.weights))
