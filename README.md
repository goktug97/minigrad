Minigrad
==========

An autograd engine inspired by Karpathy's
[micrograd](https://github.com/karpathy/micrograd) library. It is a
Drop-In Replacement but it requires you to cast Python numerical types
to the Value type.

It is called minigrad because it is a little bit bigger and more complex in terms of readability than micrograd. Engine is implemented in Python C API so it runs faster.

Currently, minigrad.nn and micrograd.nn are the same.

## Installation

``` bash
pip install minigrad
```

## Example usage

``` python
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3.0 # The power exponent must be Python double not Value.
c += c + Value(1.0)
c += Value(1.0) + c + -a
d += d * Value(2.0) + (b + a).relu()
d += Value(3.0) * d + (b - a).relu()
e = c - d
f = e**2.0
g = f / Value(2.0)
g += Value(10.0) / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

## Training a neural net
Refer to
[demo.ipynb](https://github.com/goktug97/minigrad/tree/master/examples/demo.ipynb)
example. It differs from the original implementation in terms of types
as the library only allows Value to Value operations. For example
`Value(2.0) * 2.0` is not allowed, so it should be `Value(2.0) * Value(2.0)`.

## Running tests

Should be run under test directory because It will try to import local
minigrad and fail because of the C extension. Also it requires PyTorch.

``` bash
cd test
python -m pytest
```

## License
Minigrad is licensed under the MIT License.
