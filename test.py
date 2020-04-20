from engine import Value
import numpy as np

x = Value(-4.0)
z = Value(2.0) * x + Value(2.0) + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xmg, ymg = x, y

print(xmg, ymg)

print(xmg.grad)

import torch
x = torch.Tensor([-4.0]).double()
x.requires_grad = True
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xpt, ypt = x, y
print(xpt.grad.item())



# a = np.array([1.0, 2.0, 3.0, 4.0]).astype(engine.Value)
# print(a[0]._prev)
# print(dir(value))


