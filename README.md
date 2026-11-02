# GB NET library

Welcome to gb-net! 
This was originally built for only scalars but I realised I can make it work for vectors without rewriting the whole thing. Alls well that ends well!!


## ABOUT
Gb-net is a superlightweight deep learning library I made from scratch, using python and numpy. This is my personal implementation of the knowledge shared by the goats **Andrej Karpathy** and **3Blue1Brown**. I am still inexperienced so this is only the starting line and all I have left is to improve.
If you've ever watched videos on neural nets and deep learning libraries but never understood how they work or you only know basic python, this library is definitely for you. This was made with my limited experience so it will have a lot of inefficiencies and unnecessary code so in future if anyone is reading this, all opinions are welcome.....

## Installation: 
    pip install gb-net-lib

- make sure you have numpy installed otherwise ``` pip install numpy ```

## Short usecase ( train a simple neural network )

this is a 3 layer neural network in just 6 lines of code!!!! (I've always wanted to do this. this part looks really cool)

```python
import numpy as np
import gb_net_lib

bp = gb_net_lib.bp
net = gb_net_lib.net

# 1. Make some dummy data (4 samples, each with 2 inputs, predicting 1 output)
X = [[0.1, 0.2], [0.4, 0.5], [0.7, 0.8], [0.9, 0.1]]
Y = [[0.3], [0.9], [0.98], [0.5]]

# 2. Create the net: 2 inputs -> hidden layer (4 neurons) -> hidden layer (3 neurons) -> output (1 neuron)
# Using 'sg' (sigmoid) because our targets are between 0 and 1
my_net = net(N=2, layers=[4, 3, 1], act='sg')

# 3. Train it! (trains one sample at a time)
trained_weights = my_net.train(X, Y, epochs=100, lr=0.1)

```

## How It Works (The Breakdown)

The whole library is split into two main classes:

- bp: The backpropagation engine. It wraps around NumPy arrays, keeps track of what multiplied or added with what, and calculates gradients automatically.

- net: The neural network builder. Instead of making separate Neuron and Layer classes, everything is managed in one clean class using lists of weight matrices.

### 1. The bp class (autograd engine)

Whenever you do math with bp objects, the library builds a family tree (parent list) behind the scenes so it knows how to walk backward and find the derivatives later.

```python
import gb_net_lib
import numpy as np

bp = gb_net_lib.bp
a = bp([[1.0, 2.0], [3.0, 4.0]])
b = bp([[0.5], [1.5]])
```

#### What `bp` can do:
* **Addition (`a + b`)**: Standard element-wise addition.
* **Subtraction (`a - b`)**: Standard element-wise subtraction.
* **Matrix Multiplication (`a * b`)**: **Heads up!** The `*` operator in `bp` does **dot product / matrix multiplication** (`np.dot`), *not* element-wise multiplication. Make sure your matrix shapes line up ($n \times m$ multiplied by $m \times k$)!
* **Squaring (`a.sq()`)**: Squares every element ($x^2$). Super useful for calculating squared error loss.
* **Sigmoid Activation (`a.sg()`)**: Applies the classic sigmoid squishification function:
  $$\text{sg}(x) = \frac{1}{1 + e^{-x}}$$
* **ReLU Activation (`a.relu()`)**: The classic 3Blue1Brown special. If a number is greater than $0$, it stays the same; otherwise, it becomes $0$.
* **Backpropagation (`loss.backprop()`)**: Sorts the whole computation graph in topological order and calculates the gradients for every `bp` object involved. Once you run this, check `a.grad` to see the gradients!

---


### 2. The `net` Class (The Neural Network)

The `net` class takes `bp` and turns it into a working Multi-Layer Perceptron (MLP).

#### `net(N, layers=[], act='relu')`
* **`N`** *(int)*: Number of input features your data has.
* **`layers`** *(list of ints)*: Each number is how many neurons you want in that layer. For example, `[16, 8, 1]` makes a 3-layer network with $16$ neurons in the first hidden layer, $8$ in the second, and $1$ output neuron.
* **`act`** *(str)*: Your activation function. Either `'relu'` (default) or `'sg'` (sigmoid).

#### `.layer_init()`
Generates all the random weight matrices for your network and returns them in a list. 
* Weights are initialized randomly between $-0.5$ and $0.5$.
* Every layer's weight matrix has the shape $(\text{neurons in current layer}, \text{inputs from previous layer})$.

#### `.feed(struc)`
Runs a forward pass through the network!
* You set `my_net.input` to a column vector `bp` object, pass your list of weights (`struc`) into `.feed(struc)`, and it multiplies the inputs through every layer and applies the activation function.
* Returns a list containing the output of every single layer (so `dat[-1]` is your final prediction).

#### `.train(X, Y, epochs=100, lr=0.01)`
The main training loop. 
* Uses **Stochastic Gradient Descent (one-by-one training)**—no complicated mini-batching. It looks at one data point, guesses, checks the squared error loss $(\text{pred} - \text{target})^2$, runs `backprop()`, updates the weights, resets the gradients, and moves to the next sample.
* Prints out the total loss every 10 epochs so you know it's actually learning.
* Returns the trained weight structure (`struc`) when it's done.

---
## Use Cases & Examples

### Use Case 1: Training a Network & Making Predictions Later

Once `.train()` finishes, it hands you back the trained weights (`struc`). Here's how you use those trained weights to make predictions on brand-new data:

```python
import numpy as np
import gb_net_lib

bp = gb_net_lib.bp
net = gb_net_lib.net

# Training data: Simple logic gate style problem
X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
# Let's try learning the OR gate
Y = [
    [0.0],
    [1.0],
    [1.0],
    [1.0]
]

# Initialize net with 2 inputs, 1 hidden layer of 4 neurons, and 1 output neuron
model = net(N=2, layers=[4, 1], act='sg')

# Train for 500 epochs with a slightly higher learning rate
trained_weights = model.train(X, Y, epochs=500, lr=0.5)

# --- Making a prediction on a single sample ---
test_sample = [1.0, 0.0]

# Reshape into a column vector (n x 1) just like train() does internally
model.input = bp(np.array(test_sample).reshape(-1, 1))

# Run feed-forward and grab the last layer's output
layer_outputs = model.feed(trained_weights)
prediction = layer_outputs[-1]

print("Prediction for [1, 0]:", prediction.data)
```


Because this library was built to keep things as simple and readable as possible, there are a few quirks you should keep in mind:

1. **No Biases (Yet)**: Right now, layers only do $W \cdot x$ without adding a bias vector $+ b$. *Workaround:* If you really need a bias, just append a `1.0` to the end of every input vector in `X` and increase `N` by $1$!
2. **Activation on the Final Layer**: The `.feed()` function applies your chosen activation function (`relu` or `sg`) to *every* layer, including the final output layer. 
   * If you use `act='relu'`, your network can only output numbers $\ge 0$.
   * If you use `act='sg'`, your network can only output numbers between $0$ and $1$. Make sure your target values `Y` match that range!
3. **Column Vectors Rule Everything**: Internally, `net` expects inputs to be column vectors of shape $(N, 1)$. `.train()` reshapes your lists automatically, but if you call `.feed()` manually, remember to `.reshape(-1, 1)` your input array.

---

