import math
import numpy as np

# =========================================================== the bp class for implementing backprop ==================================================================
class bp:
    # initialising the class the implements backprop
    def __init__(self, data, parent=[]):
        self.data = np.array(data, dtype=float)
        self.parent = parent
        # using zeros of the same shape instead of empty list so numpy addition works properly
        self.grad = np.zeros_like(self.data, dtype=float)
        self.back = lambda : None

    def __repr__(self):
        return (f"bp object: {self.data}")

    def __add__(self, other):
        o = bp(self.data + other.data, [self,other] )
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
        o = bp(np.dot(self.data,other.data), [self, other])
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
        o = bp( 1/( 1 + np.exp(-self.data) ), [self,] )
        def back():
            self.grad += (o.data * (1 - o.data)) * o.grad
        o.back = back
        return o

    # 2. ReLu: the second one I learnt: credit 3blue1brown
    def relu(self):
        # forgot to pass self as parent earlier so backprop wasn't going past relu
        o = bp( np.where(self.data>0,self.data, 0), [self,] )
        def back():
            self.grad += np.where(o.data>0,1,0) * o.grad
        o.back = back
        return o

    # 3 softmax : for later if i get time ====================================================================!! remember to implement !!
    def softmax(self):
        pass

    # 4. cross entropy loss also if i get time later =========================================================!! remember to implement !!
    def cel(sef):
        pass

    # the function that actually starts the backprop chain 
    def backprop(self):
        self.grad = np.ones_like(self.data, dtype=float)
        # it hold the topological order to work its way down
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

# Theory: I'm going to do some matrix calculations here using the neural net so i'm thinking should i rather make a layer class or should i start with a neuron class and build up to a layer. Last time i went ahead with the neuron building up with layer class approach this time liek keras i think i'll try making it single classes so its easier to manage.



# net class to start with neural net implementation
class net:
    
    def __init__(self, N, layers=[], act = 'relu'):
        # the layers list will hold number of neurons in it and number of elements represents number of layers
        self.layer_num = len(layers)
        self.layer_str = layers
        self.input_num = N
        self.activation = act
        self.input = bp( np.array([]))

    def layer_init(self):
        l = []
        # to avoid out of bounds error added this extra layer
        l.append(bp(np.random.rand(self.layer_str[0],self.input_num )-0.5 ))
        
        for i in range(1,self.layer_num):
            # so the thing is i want n+1 layered list to act as the parent holding all of this 
            l.append(bp(np.random.rand(self.layer_str[i],self.layer_str[i-1] )-0.5 )) # taking away 0.5 to normalise it
            
        # this makes layers along with their random weights. the list stores the layers of the network while the elements inside the list contains 2d arrays n*m where n is number of neurons and m is number of inputs to that neuron
        return l




    def feed(self, struc): 
        # now that the function for creating the layers is in place i am creating the feed forward function
        # sum of xiwi from 1 to n which would be implemented by vectors
        dat = []
        # this makes the identity to multiply into the array: every output array will be of the shape n*1 so i will have to make n*n identity matrix 
        out = self.input
        for i in struc:
            out = i*out
            # passing the layer output through the chosen activation function
            if self.activation == 'relu':
                out = out.relu()
            elif self.activation == 'sg':
                out = out.sg()
            dat.append(out)

        return dat




    def train(self, X, Y, epochs=100, lr=0.01):
        # initialising the weights before starting the training loop
        struc = self.layer_init()

        # training one sample at a time instead of batches to keep things simple
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X, Y):
                # reshaping input and target into n*1 column vectors so matrix multiplication works
                self.input = bp(np.array(x, dtype=float).reshape(-1, 1))
                target = bp(np.array(y, dtype=float).reshape(-1, 1))

                # forward pass and getting the last layer output as prediction
                dat = self.feed(struc)
                pred = dat[-1]

                # simple squared error loss (pred - target)^2
                loss = (pred - target).sq()
                total_loss += np.sum(loss.data)

                # running backprop from the loss node
                loss.backprop()

                # updating weights with gradient descent and resetting grads for the next sample
                for layer in struc:
                    layer.data -= lr * layer.grad
                    layer.grad = np.zeros_like(layer.data, dtype=float)

            if epoch % 10 == 0:
                print(f"epoch {epoch} | loss: {total_loss}")

        return struc