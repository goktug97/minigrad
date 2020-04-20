from engine import Value
import numpy as np

x = Value(-4.0)
z = Value(2.0) * x + Value(2.0) + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xmg, ymg = x, y

import torch
x = torch.Tensor([-4.0]).double()
x.requires_grad = True
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xpt, ypt = x, y

assert ymg.data == ypt.data.item()
assert xmg.grad == xpt.grad.item()

input = np.array([Value(1.0), Value(2.0)])
a = input * input
output = a[0] + a[1]
output.backward()
print(input[0].grad)



