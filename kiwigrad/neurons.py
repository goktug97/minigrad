#from .engine import Value
from kiwigrad.engine import Value
import random 
from typing import Literal
from .skeleton import Module


class Neuron(Module):

    def __init__(self, nin, bias: bool = True, activation: Literal["relu", "sigmoid", "tanh", "linear"] = "linear"):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        if bias: 
            self.b = Value(random.uniform(-1,1))
        else: 
            self.b = Value(0.)
        if activation not in ["relu", "sigmoid", "tanh", "linear"]:
            raise ValueError("Select between the following activation functions: relu, sigmoid, tanh, linear.")
        self.activation = activation

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return act.relu()
        elif self.activation == "sigmoid":
            return act.sigmoid()
        elif self.activation == "tanh":
            return act.tanh()
        elif self.activation == "linear":
            return act 
        else:
            pass

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        if self.activation == "relu":
            return f"ReLu Neuron({len(self.w)})"
        elif self.activation == "sigmoid":
            return f"Sigmoid Neuron({len(self.w)})"
        elif self.activation == "tanh":
            return f"TanH Neuron({len(self.w)})"
        elif self.activation == "linear":
            return f"Linear Neuron({len(self.w)})"
        else:
            pass

    

class RNN1Neuron(Module):
 
    def __init__(self, nin, bias: bool = True, activation: Literal["relu", "sigmoid", "tanh", "linear"] = "linear"):
        self.w_xh = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.w_hh = Value(random.uniform(-1,1)) 
        if bias: 
            self.b = Value(random.uniform(-1,1))
        else: 
            self.b = Value(0.)
        self.h = Value(0)
        if activation not in ["relu", "sigmoid", "tanh", "linear"]:
            raise ValueError("Select between the following activation functions: relu, sigmoid, tanh, linear.")
        self.activation = activation
 
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w_xh, x)), self.b) + (self.w_hh*self.h)
        if self.activation == "relu":
            self.h = act.relu()
            return act.relu()
        elif self.activation == "sigmoid":
            self.h = act.sigmoid()
            return act.sigmoid()
        elif self.activation == "tanh":
            self.h = act.tanh()
            return act.tanh()
        elif self.activation == "linear":
            self.h = act
            return act 
        else:
            pass
 
    def parameters(self):
        return self.w_xh + [self.w_hh] + [self.b]
 
    def __repr__(self):
        if self.activation == "relu":
            return f"ReLu RNN1Neuron({len(self.w)})"
        elif self.activation == "sigmoid":
            return f"Sigmoid RNN1Neuron({len(self.w)})"
        elif self.activation == "tanh":
            return f"TanH RNN1Neuron({len(self.w)})"
        elif self.activation == "linear":
            return f"Linear RNN1Neuron({len(self.w)})"
        else:
            pass
