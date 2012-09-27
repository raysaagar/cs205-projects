from mrjob.job import MRJob
import math
import sys

class mrLetterCount(MRJob):

	def mapper(self, key, word):
		for c in word:
                	if c.isalpha():
	                       	yield c.upper(), 1

        def reducer(self, char, counts):
                yield char, sum(counts)
        
if __name__ == '__main__':
        mrLetterCount.run()
