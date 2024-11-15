
import cv2
import numpy as np
import os


#Model HyperParameters

LEARNING_RATE = 0.05
EPOCHS = 10
BATCHSIZE = 32



class MaxPool:
#FINSHED

#(Channels, x,y)

    def __init__(self, input_shape, pool_size=(2, 2)):
        self.pool_shape = pool_size
        self.input_shape = input_shape
        self.output_shape = (self.input_shape[0],
                        self.input_shape[1] // self.pool_shape[0],
                        self.input_shape[2] // self.pool_shape[1])
        self.input = np.zeros(input_shape)

    def forward_pass(self,input_data):
        self.input = input_data
        output = np.empty(self.output_shape)
        for f in range(self.input.shape[0]):
            for x in range(self.output_shape[1]):
                for y in range(0,self.output_shape[1]):
                    output[f,x,y] = input_data[f,
                                         x * self.pool_shape[0]:(x+1)* self.pool_shape[0],
                                         y * self.pool_shape[1]:(y+1)* self.pool_shape[1]].max()
        return output
    def backprop(self,gradient,sample):
        # Gradient has a shape of (layer, x, y)

        # input has a shape of (layer, x * poolshape[0], y*poolshape[1])
        output_grad = np.zeros_like(self.input)

        for layer_idx,layer in enumerate(self.input):
            for x in range(gradient.shape[1]):
                for y in range(gradient.shape[2]):
                    max_index_flat = np.argmax(self.input [layer_idx,
                              x * self.pool_shape[0]: (x+1) * self.pool_shape[0],
                              y * self.pool_shape[1]: (y+1) * self.pool_shape[1]])
                    max_index_2d = np.unravel_index(max_index_flat, self.pool_shape)
                    max_global = (x * self.pool_shape[0] + max_index_2d[0],
                                  y * self.pool_shape[1] + max_index_2d[1])
                    output_grad [layer_idx,*max_global] = gradient[layer_idx,x,y]
        return output_grad
    def  apply_gradient(self):
        return

class Flatten:
    def __init__(self, input_shape):
        self.shape = input_shape
        self.output_shape = (np.prod(input_shape),)

    def forward_pass(self,input_data):
        return input_data.flatten()
    def backprop(self,gradient,sample):
        return gradient.reshape(self.shape)
    def  apply_gradient(self):
        return

class DenseLayer:
    def __init__(self, input_shape, size, activation='relu'):
        self.activation = activation
        self.input_shape = input_shape
        self.input = np.empty(input_shape)
        self.logit = np.empty((size,))
        self.output = np.empty((size,))
        fan_in = self.input.shape[0]  # number of input features per neuron
        self.weights = np.random.randn(self.output.shape[0], fan_in) * np.sqrt(2 / fan_in)
        self.bias = np.zeros(size)

        self.bias_grad = np.empty((BATCHSIZE, size))
        self.weights_grad = np.empty((BATCHSIZE, *self.weights.shape))

    def forward_pass(self, input_vector):
        # Save input for use in backpropagation
        self.input = input_vector

        for i, node in enumerate(self.weights):
            # Forward pass: z_i = x @ w_i + b_i (dot product of input and weights)
            # z_i = logit for node i in this layer
            self.logit[i] = np.sum(node * self.input) + self.bias[i]

        # Apply activation function
        self.output = activation(self.activation, self.logit)
        return self.output

    def backprop(self, gradient, sample):
        # Compute gradient of the loss with respect to logits:
        # dL/dz_i = gradient * activation_derivative(logit_i)
        self.bias_grad[sample] = gradient * derivitive(self.activation, self.logit)

        # Compute gradient of the loss with respect to weights:
        # dL/dw_i = dL/dz_i * input (broadcasted for each neuron)
        self.weights_grad[sample] = [self.input * self.bias_grad[sample][i] for i in range(self.output.shape[0])]

        # Compute gradient of the loss with respect to input for backprop to the previous layer
        input_grad = np.zeros_like(self.input)
        input_layer_weights = np.swapaxes(self.weights, 0, 1)  # shape (fan_in, output size)

        # Input gradients for each input neuron
        for i, node in enumerate(input_layer_weights):
            # Sum contributions of all downstream errors for this input node
            input_grad[i] = np.sum(node * self.bias_grad[sample])

        return input_grad

    def apply_gradient(self):
        # Update weights and biases using averaged gradients
        self.weights -= np.mean(self.weights_grad, axis=0) * LEARNING_RATE
        self.bias -= np.mean(self.bias_grad, axis=0) * LEARNING_RATE


class Model:
    def __init__(self, train_set, test_set):
        self.layers = []
        self.train_set = train_set
        self.test_set = test_set
        self.output_vector_shape = (self.train_set[0][0]).shape

    def forward_pass(self, base_input):
        input_layer = base_input
        for layer in self.layers:
            input_layer = layer.forward_pass(input_layer)
        return input_layer

    def apply_gradient(self):
        for layer in self.layers:
            layer.apply_gradient()


    def shuffle(self):
        np.random.shuffle(self.train_set)

    def test(self, label):
        # Identify class with highest activation (predicted class)
        greatest_activation = np.argmax(self.layers[-1].output)
        return greatest_activation == label

    def calculate_gradient(self, batch_set):
        
        total_samples = 0
        correctly_identified = 0

        for sample in range(len(batch_set)):
            image = batch_set[sample][0]
            label = int(batch_set[sample][1])
            output = self.forward_pass(image)
            total_samples += 1
            if self.test(label):
                correctly_identified += 1

            # Calculate errors
            desired_output = np.zeros(self.layers[-1].output.shape)
            desired_output[label] = 1
            logit_error = output - desired_output  # Gradient of cross-entropy loss w.r.t. logits for output layer

            output_layer = self.layers[-1]

            # Backpropagation through output layer:
            # dL/dz = softmax(output) - one_hot(label)
            output_layer.bias_grad[sample] = logit_error
            output_layer.weights_grad[sample] = [output_layer.input * output_layer.bias_grad[sample][i]
                                                 for i in range(output_layer.output.shape[0])]
            input_grad = np.zeros_like(output_layer.input)
            input_layer_weights = np.swapaxes(output_layer.weights, 0, 1)

            for j, node in enumerate(input_layer_weights):
                input_grad[j] = np.sum(node * output_layer.bias_grad[sample])

            for layer in self.layers[::-1][1:]:

                input_grad = layer.backprop(input_grad, sample)


        accuracy = correctly_identified / total_samples
        print(f'The Accuracy of this batch was {accuracy}')

    def train(self):
        total_full_batches = len(self.train_set) // BATCHSIZE
        remaining_samples = len(self.train_set) % BATCHSIZE

        for epoch in range(EPOCHS):
            self.shuffle()
            for i in range(total_full_batches):
                batch_set = self.train_set[i * BATCHSIZE: (i + 1) * BATCHSIZE]
                self.calculate_gradient(batch_set)
                self.apply_gradient()

            if remaining_samples != 0:
                batch_set = self.train_set[total_full_batches * BATCHSIZE:]
                self.calculate_gradient(batch_set)
                self.apply_gradient()

    def add_dense_layer(self, size, activation):
        next_layer = DenseLayer(self.output_vector_shape, size, activation)
        self.output_vector_shape = (size,)
        self.layers.append(next_layer)
    def add_max_pool(self,pool_size):
        next_layer = MaxPool(self.output_vector_shape,pool_size)
        self.output_vector_shape = next_layer.output_shape
        self.layers.append(next_layer)
    def add_flatten(self):
        next_layer = Flatten(self.output_vector_shape)
        self.output_vector_shape = next_layer.output_shape
        self.layers.append(next_layer)
    def add_conv_2d(self,num_filters,Kernal_size,activation):
        next_layer = Conv2D(self.output_vector_shape,num_filters,Kernal_size,activation)
        self.output_vector_shape = next_layer.output_shape
        self.layers.append(next_layer)


def load_data(path):
    output = []
    sub_dir_list = os.listdir(path)
    for name in sub_dir_list:
        sub_dir = os.path.join(path, name)
        if not os.path.isdir(sub_dir):
            continue

        img_dir_list = os.listdir(sub_dir)
        for img_name in img_dir_list:
            img_dir = os.path.join(sub_dir, img_name)
            image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = image.astype(np.float32) / 255.0
                # adding a channels argument to be compatible with convoution
                image = image[np.newaxis, :]
                output.append((image, name))
            else:
                print(f"Warning: Failed to load image {img_dir}")
    return output

class Conv2D:
    # EXPECTS AN INPUT SHAPE OF (Channels, X, Y)
    def __init__(self, input_shape, num_filters=32, kernel_size=(3, 3), activation='relu'):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.input= np.empty(input_shape)

        if len(input_shape) != 3 :
            raise ValueError("Sorry, Input tensors must be 3D")

        self.activation = activation

        # Initialize filters with shape (num_filters, input_channels, kernel_height, kernel_width)

        fan_in = kernel_size[0] * kernel_size[1] * input_shape[0]


        self.filters = np.random.randn(num_filters, self.input_shape[0], kernel_size[0], kernel_size[1])* np.sqrt(2 / fan_in)


        self.filters_grad = np.empty((BATCHSIZE , *self.filters.shape))

        self.bias = np.zeros((num_filters,))
        self.bias_grad = np.empty((BATCHSIZE,num_filters))

        # Calculate output dimensions
        output_height = self.input_shape[1] - kernel_size[0] + 1
        output_width = self.input_shape[2] - kernel_size[1] + 1

        self.output_shape= (num_filters, output_height, output_width)
        self.logit = np.empty(self.output_shape)

        self.input = np.empty(self.input_shape)
        self.input_grad = np.zeros_like(self.input)

    def forward_pass(self,input_tensor):
        self.input = input_tensor
        self.logit = conv2d_multi_channel(self.input,self.filters, bias = self.bias)
        return activation(self.activation,self.logit)

    def backprop(self, input_gradient, sample):

        logit_partial = input_gradient * derivitive(self.activation,self.logit)


        #weights
        #input shape is cin,x,y output = cout,x,y
        logit_partial_copy = np.copy(logit_partial)
        self.filters_grad[sample] = conv2d_multi_channel(self.input,logit_partial_copy)

        #biases
        self.bias_grad[sample] = np.sum(logit_partial, axis=(1, 2))

        #input layer
        rot_filter = self.filters[:,:,::-1,::-1]

        return  conv2d_multi_channel(logit_partial,rot_filter, mode = 'full')

    def apply_gradient(self):
        self.filters -= np.mean(self.filters_grad, axis=0) * LEARNING_RATE
        self.bias -= np.mean(self.bias_grad, axis=0) * LEARNING_RATE

def derivitive(activation, x):
    if activation == 'relu':
        return np.where(x > 0, 1, 0)
    elif activation == 'softmax':
        pass
    elif activation == 'sigmoid':
        pass


def activation(activation, x):
    if activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'softmax':
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))


def conv2d_multi_channel(input_tensor, kernels, bias=None, mode='valid'):

    C_in, H, W = input_tensor.shape

    flag = False

    if len(kernels.shape) == 3:
        flag= True
        C_out, K_h, K_w = kernels.shape

        kernels = np.expand_dims(kernels,axis=0)
        #print(f' Rotation {kernels.shape}')

       # print(f'Logit partial{input_tensor.shape}')

    else:
        C_out, _, K_h, K_w = kernels.shape

    # Determine padding based on the mode
    if mode == 'full':
        pad_h = K_h - 1
        pad_w = K_w - 1
        # Pad input tensor with zeros
        input_tensor = np.pad(input_tensor, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        H += 2 * pad_h
        W += 2 * pad_w
    elif mode != 'valid':
        raise ValueError("Invalid mode. Supported modes are 'valid' and 'full'.")

    # Calculate output dimensions
    H_out = H - K_h + 1
    W_out = W - K_w + 1

    # Initialize output tensor
    output_tensor = np.zeros((C_out, H_out, W_out))

    # Perform convolution for each output channel
    for c_out in range(C_out):
        # Initialize the output for this channel
        for c_in in range(C_in):
            # Slide the kernel over the input feature map
            for h in range(H_out):
                for w in range(W_out):
                    # Compute the convolution at this position
                    if kernels.shape[0] != 1 and kernels.shape[1] != 1:
                        output_tensor[c_out, h, w] += np.sum(
                            input_tensor[c_in, h:h + K_h, w:w + K_w] * kernels[c_out, c_in, :, :]
                        )
                    elif kernels.shape[0] == 1:
                        output_tensor[c_out, h, w] += np.sum(
                            input_tensor[c_in, h:h + K_h, w:w + K_w] * kernels[0, c_in, :, :])
                    else:
                        output_tensor[c_out, h, w] += np.sum(
                            input_tensor[c_in, h:h + K_h, w:w + K_w] * kernels[c_out, 0, :, :])

        # Add bias if provided
        if bias is not None:
            output_tensor[c_out, :, :] += bias[c_out]

    if flag and mode == 'valid':
        output_tensor= np.expand_dims(output_tensor,axis=1)


    return output_tensor

if __name__ == '__main__':
    training_dir = os.path.join('mnist_png', 'training')
    testing_dir = os.path.join('mnist_png', 'testing')

    train_set = load_data(training_dir)
    test_set = load_data(testing_dir)
    print("Data loaded")

    slowest_model_ever = Model(train_set, test_set)

    # Adding layers to the model
    slowest_model_ever.add_flatten()
    slowest_model_ever.add_dense_layer(64, 'relu')
    slowest_model_ever.add_dense_layer(10, 'softmax')

    # Start training
    slowest_model_ever.train()

