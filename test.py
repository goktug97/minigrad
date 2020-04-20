import engine
import numpy as np

p1 = engine.Value(20.0)
p2 = engine.Value(20.0)
a = p1.relu()
print(hash(p1))
# a = p1/p2
print('a')
print(a.data)
a.grad = -30
print(p1.grad)
a.backward()
print(p1.grad)
# print(p1.grad)
# print(a._prev)



# a = np.array([1.0, 2.0, 3.0, 4.0]).astype(engine.Value)
# print(a[0]._prev)
# print(dir(value))


