import engine


p1 = engine.Value(20.0)
p2 = engine.Value(20.0)
value = engine.Value(20.0, (p1,p2))
# print(p1._prev)
# print(p2._prev)
# print(p1._prev)
# print(p2._prev)
# print(value._prev)
value = p1 + p2
value.grad = 30
# print(value._prev)
print(value._backward())
print(p1.grad)
print(p2.grad)

# print(dir(value))


