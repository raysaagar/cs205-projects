from mrjob.job import MRJob
import math
import sys

INTERVALS = 100001 # number of intervals
h = 1.0/INTERVALS # step size
a = 0 # start of interval
b = 1 # end of interval

# circle function
def f(x):
	return math.sqrt(1 - pow(float(x),2))

class mrTrapInt(MRJob):

	# mapper yields value calculated for each step
	def mapper(self, key, value):
		yield "pi", f(float(value)*h)

	# reducer combines results from mapper with the end points
	def reducer(self, key, values):
		yield key, ((f(a) + f(b))/2 + sum(values))*h

if __name__ == '__main__':
	mrTrapInt.run()
