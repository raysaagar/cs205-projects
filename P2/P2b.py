from mrjob.job import MRJob
import math
import sys

INTERVALS = 100001 # number of intervals
h = 1.0/INTERVALS # step size
a = 0 # start
b = 1 # end

# circle function
def f(x):
	return math.sqrt(1 - pow(float(x),2))

class mrTrapInt(MRJob):

	# initializer
	def __init__(self, *args, **kwargs):
	        super(mrTrapInt, self).__init__(*args, **kwargs)
		self.sum = 0

	# store sum in self variable. yield required as convention of MRJob
	def mapper(self, key, value):
		if False:
			yield
		self.sum += f(float(value)*h)
	
	# in mapper combiner to yield the final sum
	def mapper_final(self):
		yield "pi", self.sum

	# reducer combines values from mapper with end points
	def reducer(self, key, values):
		yield key, ((f(a) + f(b))/2 + sum(values))*h

if __name__ == '__main__':
	mrTrapInt.run()
