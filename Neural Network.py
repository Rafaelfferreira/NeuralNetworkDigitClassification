import numpy as np
from random import shuffle

#sigmoid function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#derivative of the sigmoid function
def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))

#Main class that represents the network
class Network(object):
    def __init__(self, sizes): #sizes is a list that contains the number of neurons in each respective layer
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] #randomly initializing the biases afther the firs layer (because the first one is the input layer)
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])] #randomly initializing the weights
        
    def feedForward(self, a):
        "Return the output of the network if 'a' is input."
        for b, w in zip(self.biases, self.weights): #zip aggregates iterables into tuples
            a = sigmoid(np.dot(w,a)+b) #np.dot = produto vetorial
            
    def SGD(self, training_data, epochs, miniBatchSize, eta, test_data = None): #Stochastic gradient descent
        #eta is the learning rate n
        #"Train the neural network using mini-batch stochastic gradient descent. the training_data is a list of tuples representing the training inputs and the desired outputs."
        #"If test_data is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out. Thi is useful for tracking progress but it slows things down substantialy"
        if test_data: nTest = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data) #shuffle the order of training_data
            #now that the training_data has been shuffled we will partition it in miniBatches
            miniBatches = [training_data[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)] #This range goes from 0 to n in miniBatchSize jumps
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta) #calls a function that updates the weights and biases of the network according to a single iteration of gradient descent useing just the data in the miniBatch
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), nTest))
            else:
                print("Epoch {0} complete".format(j))
                
    #this function updates the weights and biases of the network by computing the gradient of the current mini batch
    def updateMiniBatch(self, miniBatch, eta):
        #"Update the network's weights and biases applying gradient descent using backpropagation to a single mini batch"
        #The 'miniBatch' is a list of tuples and 'eta' is the learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases] #initializing gradient of biases with all zeroes
        nabla_w = [np.zeros(w.shape) for w in self.weights] #gradient of weights
        for x,y in miniBatch: #computes the gradient for every training example on miniBatch
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) #invokes the backpropagation algorithm and returns a new gradient vector for w and b
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #updates the direction of the gradient vector
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(miniBatch))*nw for w,nw in zip(self.weights, nabla_w)] #updates the weights
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self,x ,y):
        #"Return a tuple (nabla_b, nabla_w) representing the gradient cost function C(x)"
        #nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
        nabla_b = [np.zeros(b.shape) for b in self.biases] # .shape method takes the dimension of the object that called it
        nabla_w = [np.zeros(w.shape) for w in self.weights] #np.zeros(w.shape) basically means "instantiate an array of zeroes with the dimensions of 'w'"
        #feedforward
        activation = x #x is the activation value of each neuron
        activations = [x] #list to store all the activations, layer by layer
        zs = [] #list to store all the z vectors, layer by layer. Z vector is the value of 'w.a + b'
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b #this is what goes inside the sigmoid function
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1]) #negative indexes are suppose to be counted from the right to the left, ie, -1 is the last element of the array
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #-2 is the second last element of the array
        #The variable 'l' on the loop below is to be interpreted as follow:
        #l = 1 means the last layer, l = 2 is the second last layer and so on.
        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self,test_data):
        # Returns the number of test inputs for which the neural network outputs the correct result.
        testResults = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in testResults)
    
    def costDerivative(self, outputActivations, y):
        # Return the vector of partial derivatives d(C(x))/d(a)
        return (outputActivations-y)