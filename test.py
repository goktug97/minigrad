import engine


p1 = engine.Value(20.0)
p2 = engine.Value(20.0)
value = engine.Value(20.0, (p1,p2))
print(value._prev)

# print(dir(value))


