def add1(x):
	return x+1

add2 = lambda x: x+2

def functor_add(x):
	return lambda y: y+x

add3 = functor_add(3)
add4 = functor_add(4)

print(add1(1)) # Prints 2
print(add2(1)) # Prints 3
print(add3(1)) # Prints 4
print(add4(1)) # Prints 5
