## Kiwigrad

<h1 align="center">
<img src="logo.png" width="200">
</h1><br>

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 

*Despite lacking the ability to fly through the skies like **PyTorch** and **TensorFlow**, the Kiwigrad is still a formidable bird that is teeming with untapped potential waiting to be uncovered.* :wink:

- [Kiwigrad](#kiwigrad)
- [Install](#install)
- [Functionalities](#functionalities)
  - [Examples](#examples)
- [Running test](#running-test)

Kiwigrad? yes, it is another version of [micrograd](https://github.com/karpathy/micrograd). It is a Drop-In Replacement but it requires you to cast Python numerical types to the Value type.

## Install 

To install the current release,

```console
pip install kiwigrad
```

## Functionalities 

Kiwigrad is a modified version of the [micrograd](https://github.com/karpathy/micrograd) package with additional features. The main features added to Kiwigrad are:

* Training is faster due to the C implementation of the Value object.
* Tracing functionalities like the original [micrograd](https://github.com/karpathy/micrograd) package were added. An example of this can be seen in the [ops](examples/ops.ipynb) notebook.
* Methods for saving and loading the weights of a trained model.
* Support for RNN(1) feedforward neural networks.

### Examples

* In the [examples](examples/) folder, you can find examples of models trained using the Kiwigrad library.
* A declaration example of an MLP net using Kiwigrad:
  
```python 
from kiwigrad import MLP, Layer

class PotNet(MLP):
    def __init__(self):
        layers = [
            Layer(nin=2, nout=16, bias=True, activation="relu"),
            Layer(nin=16, nout=16, bias=True, activation="relu"),
            Layer(nin=16, nout=1, bias=True, activation="linear")
        ]
        super().__init__(layers=layers)

model = PotNet()
```
* Kiwigrad like [micrograd](https://github.com/karpathy/micrograd) comes with support for a number of possible operations:

```python 
from kiwigrad import Value, draw_dot

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + Value(1.)
c += Value(1.) + c + (-a)
d += d * Value(2) + (b + a).relu()
d += Value(3.) * d + (b - a).relu()
e = c - d
f = e**2
g = f / Value(2.0)
g += Value(10.0) / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

draw_dot(g)
```

## Running test

```console
cd test 
pytest .
```