import math
import numpy as np

# =========================================================== the bp class for implementing backprop ==================================================================
class bp:
    # initialising the class that implements backprop
    def __init__(self, data, parent=[]):
        self.data = np.array(data, dtype=float)
        self.parent = parent
        # using zeros of the same shape instead of empty list so numpy addition works properly
        self.grad = np.zeros_like(self.data, dtype=float)
        self.back = lambda : None

    def __repr__(self):
        return (f"bp object: {self.data}")

    def __add__(self, other):
        o = bp(self.data + other.data, [self, other])
        def back():
            self.grad += o.grad
            other.grad += o.grad

        o.back = back
        return o

    # needed subtraction and squaring to calculate simple squared error loss for training
    def __sub__(self, other):
        o = bp(self.data - other.data, [self, other])
        def back():
            self.grad += o.grad
            other.grad -= o.grad
        o.back = back
        return o

    def sq(self):
        o = bp(self.data ** 2, [self,])
        def back():
            self.grad += 2 * self.data * o.grad
        o.back = back
        return o

    def __mul__(self, other):
        o = bp(np.dot(self.data, other.data), [self, other])
        def back():
            # had to use dot product with transpose here because we are multiplying matrices/vectors
            self.grad += np.dot(o.grad, other.data.T)
            other.grad += np.dot(self.data.T, o.grad)
        o.back = back
        return o

    # defining the few activation functions
    # 1. Sigmoid: because it was the first one I learnt
    #       sg(x) = 1/[ 1 + e^(-x) ]
    def sg(self):
        o = bp(1 / (1 + np.exp(-self.data)), [self,])
        def back():
            self.grad += (o.data * (1 - o.data)) * o.grad
        o.back = back
        return o

    # 2. ReLu: the second one I learnt: credit 3blue1brown
    def relu(self):
        # forgot to pass self as parent earlier so backprop wasn't going past relu
        o = bp(np.where(self.data > 0, self.data, 0), [self,])
        def back():
            self.grad += np.where(o.data > 0, 1, 0) * o.grad
        o.back = back
        return o

    # 3. Softmax: turns raw scores (logits) into probabilities that add up to 1
    # subtracting max(x) first so exp() doesn't blow up to infinity with big numbers
    def softmax(self):
        shifted = self.data - np.max(self.data, axis=0, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=0, keepdims=True)
        o = bp(probs, [self,])
        def back():
            # Jacobian of softmax is diag(s) - s @ s.T
            # multiplying that by o.grad simplifies nicely to s * (o.grad - sum(s * o.grad))
            dot = np.sum(o.data * o.grad, axis=0, keepdims=True)
            self.grad += o.data * (o.grad - dot)
        o.back = back
        return o

    # 4. Cross Entropy Loss (cel): way better than squared error when working with probabilities
    # formula: L = -sum( target * log(pred) )
    def cel(self, target):
        # if someone passes a raw list or numpy array instead of a bp object, wrap it up
        target_bp = target if isinstance(target, bp) else bp(target)
        # clipping values slightly away from 0 so np.log(0) doesn't give NaN or -inf
        eps = 1e-15
        clipped_pred = np.clip(self.data, eps, 1.0)
        o = bp(-target_bp.data * np.log(clipped_pred), [self, target_bp])
        def back():
            # derivative of -y * log(p) with respect to p is -y / p
            self.grad += -(target_bp.data / clipped_pred) * o.grad
            target_bp.grad += -np.log(clipped_pred) * o.grad
        o.back = back
        return o

    # the function that actually starts the backprop chain 
    def backprop(self):
        # setting the starting node's gradient to 1s (dL/dL = 1)
        # note: intermediate node gradients aren't cleared here, so calling backprop() twice on the same graph will stack grads!
        self.grad = np.ones_like(self.data, dtype=float)
        # it holds the topological order to work its way down
        topo = []
        visited = []
        def build_topo(x):
            if x not in visited:
                visited.append(x)
                for i in x.parent:
                    build_topo(i)
                topo.append(x)

        build_topo(self)
        for i in reversed(topo):
            i.back()

# okay so this is made to handle vectors too
# ======================================================================================================================================================
# ======================================================================================================================================================

# Theory: I'm going to do some matrix calculations here using the neural net so i'm thinking should i rather make a layer class or should i start with a neuron class and build up to a layer. Last time i went ahead with the neuron building up with layer class approach this time like keras i think i'll try making it single classes so its easier to manage.


# net class to start with neural net implementation
class net:
    
    def __init__(self, N, layers=[], act='relu', loss='sq'):
        # the layers list will hold number of neurons in it and number of elements represents number of layers
        self.layer_num = len(layers)
        self.layer_str = layers
        self.input_num = N
        self.activation = act
        self.loss_fn = loss
        self.input = bp(np.array([]))
        # storing biases in the net object too so every layer can do (W * x) + b
        self.biases = []

    def layer_init(self):
        l = []
        b = []
        # to avoid out of bounds error added this extra layer
        l.append(bp(np.random.rand(self.layer_str[0], self.input_num) - 0.5))
        # each neuron gets its own bias in a (neurons x 1) column vector so it lines up with (W * x)
        b.append(bp(np.random.rand(self.layer_str[0], 1) - 0.5))
        
        for i in range(1, self.layer_num):
            # so the thing is i want n+1 layered list to act as the parent holding all of this 
            l.append(bp(np.random.rand(self.layer_str[i], self.layer_str[i-1]) - 0.5)) # taking away 0.5 to normalise it
            b.append(bp(np.random.rand(self.layer_str[i], 1) - 0.5))
            
        self.biases = b
        # this makes layers along with their random weights (and saves the biases in self.biases).
        # the list stores the layers of the network while the elements inside the list contain 2d arrays n*m where n is number of neurons and m is number of inputs to that neuron
        return l

    # separate helper in case we ever want to reset or grab fresh biases on their own
    def bias_init(self):
        self.biases = [bp(np.random.rand(n_neurons, 1) - 0.5) for n_neurons in self.layer_str]
        return self.biases

    def feed(self, struc, biases=None): 
        # now that the function for creating the layers is in place i am creating the feed forward function
        # computes (W * x) + b for each layer using column vectors
        dat = []
        # if biases weren't passed in manually, use the ones stored on self (or make them if layer_init wasn't called)
        if biases is None:
            if len(self.biases) != len(struc):
                self.biases = [bp(np.zeros((w.data.shape[0], 1))) for w in struc]
            biases = self.biases

        out = self.input
        for idx, (w, b) in enumerate(zip(struc, biases)):
            # linear step: matrix multiply weights with input vector, then add the bias column vector
            out = (w * out) + b
            # passing the layer output through the chosen activation function
            # if softmax is chosen, we usually want it on the final layer (or whichever layer is specified)
            if self.activation == 'relu':
                out = out.relu()
            elif self.activation == 'sg':
                out = out.sg()
            elif self.activation == 'softmax':
                # for softmax networks, hidden layers use relu so gradients flow, and the last layer uses softmax
                if idx == len(struc) - 1:
                    out = out.softmax()
                else:
                    out = out.relu()
            dat.append(out)

        return dat

    def train(self, X, Y, epochs=100, lr=0.01):
        # initialising the weights (and biases inside self.biases) before starting the training loop
        struc = self.layer_init()

        # training one sample at a time instead of batches to keep things simple
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X, Y):
                # reshaping input and target into n*1 column vectors so matrix multiplication works
                self.input = bp(np.array(x, dtype=float).reshape(-1, 1))
                target = bp(np.array(y, dtype=float).reshape(-1, 1))

                # forward pass and getting the last layer output as prediction
                dat = self.feed(struc, self.biases)
                pred = dat[-1]

                # choosing between cross entropy loss and simple squared error loss (pred - target)^2
                if self.loss_fn == 'cel' or self.activation == 'softmax':
                    loss = pred.cel(target)
                else:
                    loss = (pred - target).sq()
                total_loss += np.sum(loss.data)

                # running backprop from the loss node
                loss.backprop()

                # updating both weights and biases with gradient descent and resetting grads for the next sample
                for w, b in zip(struc, self.biases):
                    w.data -= lr * w.grad
                    w.grad = np.zeros_like(w.data, dtype=float)
                    b.data -= lr * b.grad
                    b.grad = np.zeros_like(b.data, dtype=float)

            if epoch % 10 == 0:
                print(f"epoch {epoch} | loss: {total_loss}")

        return struc