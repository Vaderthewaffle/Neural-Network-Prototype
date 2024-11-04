"""
This file contains functions and class implemetiaons for a
node, layer, and network classes
"""
from operator import truediv

import cv2
import numpy as np
import os
import sys
import random


def derivitive(activation, x):
    if activation == 'relu':
        if x > 0:
            return 1
        else:
            return 0
    elif activation == 'softmax':
        pass
    elif activation == 'sigmoid':
        pass


def activation(activation , x):
    """
    :param activation:
    :param x: a list of pre actvation values for a layer
    :return: a list of accurate node values after the activation function
    """

    if activation == 'relu':
        return np.maximum(0 , x)
    elif activation == 'softmax':
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)
    elif activation == 'sigmoid':
        return 1/(1+np.exp(-x))


class Node:

    def __init__(self, activation = 'relu', previous_layer_size = 0):
        self.activation = activation
        self.weights = np.random.rand(previous_layer_size)
        self.bias =  np.random.rand()
        self.gradient = {'Weights':[],
                         'Bias':[]
                         }
        self.logit = 0
        self.value = 0

    def apply_gradient(self, learn_rate):
        # Compute mean gradients for weights and scalar bias
        weight_gradient_means = np.mean(np.array(self.gradient['Weights']), axis=0)
        bias_gradient_mean = np.mean(np.array(self.gradient['Bias']))

        # Update weights and bias
        self.weights -= learn_rate * weight_gradient_means
        self.bias -= learn_rate * bias_gradient_mean

        # Clear gradients after update
        self.gradient = {'Weights': [], 'Bias': []}


    def forward_pass(self, previous_layer):
        value = 0
        for index, weight in enumerate(self.weights):
            value += weight * previous_layer.nodes[index].value
        value += self.bias
        self.logit = value
        return value

class DenseLayer:
    def __init__(self, activation = 'relu', size = 0, last_layer_size = 0):
        self.size = size
        self.activation = activation
        self.nodes = [
            Node(activation = activation,
                 previous_layer_size = last_layer_size)
            for i in range (size)]

    def forward_pass(self, prev_layer):
        layer_values_before_activation = [node.forward_pass(prev_layer) for node in self.nodes]
        layer_values = activation(self.activation, layer_values_before_activation)
        for i, node in enumerate(self.nodes):
            node.value = layer_values[i]


class Model:
    def __init__(self, learn_rate, epochs, batch_size, train_set, test_set):
        first_layer = DenseLayer(size=len(train_set[0][0]))

        self.layers = [first_layer]
        self.train_set = train_set
        self.test_set= test_set
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate

    def add_layer(self, LayerSize, activation = 'relu'):
        last_layer_size = self.layers[-1].size
        self.layers.append(
            DenseLayer(activation= activation ,
                       size = LayerSize,
                       last_layer_size= last_layer_size)
        )

    def train(self):
        total_full_batches = len(self.train_set) // self.batch_size
        remaining_samples = len(self.train_set) % self.batch_size
        for epoch in range(self.epochs):
            self.shuffle()
            for i in range(total_full_batches):
                print(total_full_batches-i)
                batch_set = self.train_set[ i * self.batch_size : (i+1) * self.batch_size]
                self.calculate_gradient(batch_set)
                self.apply_gradient()

            if remaining_samples != 0:
                batch_set = self.train_set[total_full_batches * self.batch_size:]
                self.calculate_gradient(batch_set)
                self.apply_gradient()


    def shuffle(self):
        random.shuffle(self.train_set)
    def test(self,label):
        greatest_activation = self.layers[-1].nodes[0].value
        greatest_activation_index = 0
        for index , node in enumerate(self.layers[-1].nodes[1:]):
            if node.value > greatest_activation:
                greatest_activation = node.value
                greatest_activation_index = index

        if  greatest_activation_index == label:
            return True
        return False

    def calculate_gradient(self, batch):
        total_samples = 0
        correctly_identified = 0
        for sample in batch:
            data = sample[0]
            label = int(sample[1])
            self.forward_pass(data)

            total_samples += 1
            if self.test(label):
                correctly_identified+=1

            #loss using cross entropy
            #loss = -np.log(self.layers[-1].nodes[label].value)

            # loss is the dot product of the output neuron and the predicted value

            # calculate logit loss for each node in the output node
            for i, node in enumerate(self.layers[-1].nodes):
                node.gradient['Bias'].append( node.value - (1 if i == label else 0) )

            #lets propogate these logit gradients backwards baby
            for i in range(len(self.layers)-2,0,-1):
                for node_idx, node in enumerate(self.layers[i].nodes):
                    sum = 0
                    for  next_layer_node in self.layers[i+1].nodes:
                        # Calculate the product
                        product = next_layer_node.weights[node_idx] * next_layer_node.gradient['Bias'][-1]


                        sum += product


                    logit_partial = sum * derivitive(self.layers[i].activation, node.logit)

                    node.gradient['Bias'].append(logit_partial)


            for layer_idx, layer in enumerate(self.layers):
                for node in layer.nodes:
                    weight_gradient = []
                    for weight in range(len(node.weights)):
                        weight_gradient.append(node.gradient['Bias'][-1] * self.layers[layer_idx-1].nodes[weight].value)
                    node.gradient['Weights'].append(weight_gradient)
        accuracy = correctly_identified/total_samples
        print(f'The Accuracy of this batch was {accuracy}')

    def apply_gradient(self):
        for layer in self.layers[1:]:
            for node in layer.nodes:
                node.apply_gradient(self.learn_rate)

    def forward_pass(self,data):
        #populate the input layer
        for i , node in enumerate(self.layers[0].nodes):
            node.value = data[i]

        #loop over all layers one after the other and propogate values through
        for i in range(1,len(self.layers)):
            self.layers[i].forward_pass(self.layers[i-1])

def load_data(path):
    output=[]

    sub_dir_list = os.listdir(path)
    for name in sub_dir_list:
        sub_dir = os.path.join(path, name)
        if not os.path.isdir(sub_dir):
            continue  # Skip non-directory files

        img_dir_list = os.listdir(sub_dir)
        for img_name in img_dir_list:
            img_dir = os.path.join(sub_dir, img_name)
            # Read the image in grayscale mode
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = image.flatten().astype(np.float32) / 255.0
                output.append((image,name))
            else:
                print(f"Warning: Failed to load image {img_dir}")
    return output


if __name__ == '__main__':
    #if len(sys.argv) == 2:
    training_dir = os.path.join('mnist_png', 'training')
    testing_dir = os.path.join('mnist_png', 'testing')

    train_set  = load_data(training_dir)
    test_set = load_data(testing_dir)
    print("Data loaded")

    network = Model(
        learn_rate = 0.1,
        epochs = 5,
        batch_size = 32,
        train_set = train_set,
        test_set = test_set
                )

    network.add_layer(32,'relu')
    network.add_layer(32, 'relu')
    network.add_layer(32, 'relu')

    #output layer
    network.add_layer(10, 'softmax')
    network.train()

