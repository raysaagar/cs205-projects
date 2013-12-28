from mrjob.job import MRJob
import math
import sys
import random

# circle function
def f(x):
	return math.sqrt(1 - pow(float(x),2))

class mrMonteCarlo(MRJob):

	# initialization, contains self numerator/denominator
	def __init__(self, *args, **kwargs):
		super(mrMonteCarlo, self).__init__(*args,**kwargs)
		self.num = 0 # number of points in the area
		self.trials = 0 # number of total points

	# mapper yields 1000 random points for each value
	# mapper adds to numerator and denominator for each trial
	def mapper(self, key, value):
		if False:
			yield
		random.seed(value)
		for i in range(1000):
			x = random.random()
			y = random.random()
			if (y < f(x)):
				self.num += 1
			self.trials += 1
	
	# yields numerator/denominator list
	def mapper_final(self):
		yield "area", [self.num,self.trials]

	# combines the mapped lists together for final result
	def reducer(self, key, values):
		finalnum = 0
		finaldenom = 0
		for(x,y) in values:
			finalnum += x
			finaldenom += y
		yield key, {'numerator':finalnum, 'denominator':finaldenom}

if __name__ == '__main__':
	mrMonteCarlo.run()
