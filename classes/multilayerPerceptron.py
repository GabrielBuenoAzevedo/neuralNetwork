import numpy as np

np.random.seed(45)

class MultilayerPerceptron:
  def __init__(self, sizes, functions, coded_result, lr, num_epochs):
    self.biases = [ np.random.randn(size) for size in sizes[1:] ]
    self.weights = [ np.random.randn(size, last_size) for (last_size, size) in zip(sizes, sizes[1:]) ]
    self.functions = functions
    self.coded_result = coded_result
    self.lr = lr
    self.num_epochs = num_epochs
    print(sizes)
    for weights in self.weights: 
      print(weights.shape)

  def training(self, training_set, training_results):
    #Make sure training_set and training_results are numpy arrays.
    training_set = np.array(training_set)
    training_results = np.array(training_results)
    #Loop thought every epoch
    for epoch in range(0,self.num_epochs):
      #Loop throught every training example
      for net_input, result in zip(training_set, training_results):
        net_input = np.array(net_input)
        result = np.array(result)
        #FeedForward
        feed_forward_result, weighted_sums, activations = self.feedForward(net_input)
        #Backpropagation
        error =  result - feed_forward_result #layer_input - result
        # print('Probabilities: ', feed_forward_result, '  -- result: ', result, '   -- error: ', error)
        errors_bias = []
        errors_weights = []
        #Last Layer
        last_activation = np.array([activations[-2]])
        error = error.reshape((-1,1))
        error_bias = error.T[0]
        error_weights = np.dot(error, last_activation)
        errors_weights.append(error_weights)
        errors_bias.append(error_bias)
        #Subsequent layers
        for i in range(2, len(self.weights)+1):
          activation_derivative = self.chooseFunction(self.functions[-i], weighted_sums[-i], True)
          new_error = np.dot(self.weights[-i+1].T, error)
          new_error = (new_error.T * activation_derivative).T
          error = new_error
          last_activation = np.array([activations[-i-1]])
          error_bias = error.T[0]
          error_weights = np.dot(error, last_activation)
          errors_bias.insert(0,error_bias)
          errors_weights.insert(0,error_weights)
        #New weights and biases
        new_weights = self.weights 
        new_biases = self.biases
        errors_weights_lr = [ error * self.lr for error in errors_weights ]
        errors_biases_lr = [ error * self.lr for error in errors_bias ]
        new_weights = np.add(new_weights, errors_weights_lr)
        new_biases = np.add(new_biases, errors_biases_lr)
        self.weights = new_weights
        self.biases = new_biases

  def feedForward(self, net_input):
    layer_input = np.array(net_input)
    activations = [net_input]
    weighted_sums = []
    for weights, biases, function in zip(self.weights, self.biases, self.functions):
      layer_input = layer_input.dot(weights.T) + biases
      weighted_sums.append(layer_input)
      activation = self.chooseFunction(function, layer_input, False)
      activations.append(activation)
      layer_input = activation
    #Softmax
    layer_input = np.exp(layer_input)
    layer_input = layer_input/np.sum(layer_input)
    return layer_input, weighted_sums , activations

  def predict(self, net_input):
    layer_input = self.feedForward(net_input)[0]
    max_index = np.argmax(layer_input)
    result = self.coded_result[max_index]
    return result

  def chooseFunction(self, function, weighted_sum, derivative):
    if function == 'sigmoid':
      if derivative == False:
        print('Sigmoid function should be here!!')
      else:
        print('Sigmoid derivative function should be here!!')
    # elif function == 'softmax':
      # exp = np.exp(weighted_sum)
      # return exp/np.sum(exp)
    return weighted_sum

  #Cost function using crossEntropy
  def costFunction(self, inputs):
    exp = np.exp(inputs)
    return exp/np.sum(exp)

# teste = MultilayerPerceptron([4,3], ['','4'])
# teste.training([ [1, 2, 1.5, 2.5] ], [ [0,0,1]])
# teste.training([ [1, 1, 1], [2,2,2], [3,3,3], [4,4,4], [5,5,5] ], [ [1,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ])

