#from .engine import Value
from kiwigrad.engine import Value
from .neurons import Neuron, RNN1Neuron
from typing import Literal
from .skeleton import Module


class Layer(Module):

    def __init__(self, nin, nout, bias: bool = True, activation: Literal["relu", "sigmoid", "tanh", "linear"] = "linear"):
        self.neurons = [Neuron(nin, bias=bias, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

class RNN1Layer(Module):

    def __init__(self, nin, nout, activation: Literal["relu", "sigmoid", "tanh", "linear"] = "linear"):
        self.neurons = [RNN1Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"RNN1 Layer of [{', '.join(str(n) for n in self.neurons)}]"