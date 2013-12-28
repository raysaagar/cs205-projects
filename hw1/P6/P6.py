from mrjob.job import MRJob
import math
import sys

class mrLetterCount(MRJob):
	
	# mapper yields tuples of (letter, 1) for each letter found
	def mapper(self, key, word):
		for c in word:
                	if c.isalpha():
	                       	yield c.upper(), 1
	# reducer sums up the counts for each letter
        def reducer(self, char, counts):
                yield char, sum(counts)
        
if __name__ == '__main__':
        mrLetterCount.run()
