from classes.multilayerPerceptron import MultilayerPerceptron
import random

#Encode output to match the three neuron scheme in the neural network
def encodeOutput(output):
  if output == 'setosa':
    return [1,0,0]
  elif output == 'versicolor':
    return [0,1,0]
  elif output == 'virginica':
    return [0,0,1]

#Load iris dataset
file = open('./data/iris.data', 'r')
file_lines = file.readlines()
iris_dataset = []
for line in file_lines:
  line = line.replace('\n', '')
  line = line.split(',Iris-')
  inputs = [ float(input) for input in line[0].split(',') ]
  output = encodeOutput(line[-1])
  entry = [inputs, output]
  iris_dataset.append(entry)
file.close()

#Loop to train and test the network multiple times
# for i in range(0,25):

#Get each set from the dataset
setosa = iris_dataset[0:50]
versicolor = iris_dataset[50:100]
virginica = iris_dataset[100:150]
#Shuffle entries
random.shuffle(setosa)
random.shuffle(versicolor)
random.shuffle(virginica)
#Get training (70%), validation(14%) and tests(16%) sets
training = setosa[0:35] + versicolor[0:35] + virginica[0:35]
validation = setosa[35:42] + versicolor[35:42] + virginica[35:42]
tests = setosa[42:50] + versicolor[42:50] + virginica[42:50]
#Shuffle sets
random.shuffle(training)
random.shuffle(validation)
random.shuffle(tests)
#Get inputs and outputs for training the network
inputs = []
outputs = []
for entry in training:
  inputs.append(entry[0])
  outputs.append(entry[1])
#Creating the network
neural_net = MultilayerPerceptron([4,3,5,2], ['',''], ['setosa', 'versicolor', 'virginica'], 0.004, 300)
# neural_net.training( inputs, outputs )
# #Testing the network
# correct_guesses = 0
# incorrect_guesses = 0
# for entry in tests:
#   net_input = entry[0]
#   expected_output = entry[1]
#   #Decoding expected output
#   if expected_output[0] == 1:
#     expected_output = 'setosa'
#   elif expected_output[1] == 1:
#     expected_output = 'versicolor'
#   else:
#     expected_output = 'virginica'
#   net_output = neural_net.predict(net_input)
#   if net_output == expected_output:
#     correct_guesses += 1
#   else:
#     incorrect_guesses += 1
# print('Correct: ', correct_guesses, ' -- Incorrect: ', incorrect_guesses)