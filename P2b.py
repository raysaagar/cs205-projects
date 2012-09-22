from mrjob.job import MRJob
import math
import sys

INTERVALS = 100001
h = 1.0/INTERVALS
a = 0
b = 1

def f(x):
	return math.sqrt(1 - pow(float(x),2))

class mrTrapInt(MRJob):

	def mapper(self, key, value):
		yield "pi", f(float(value)*h)

	def reducer(self, key, values):
		yield key, ((f(a) + f(b))/2 + sum(values))*h

if __name__ == '__main__':
	mrTrapInt.run()
