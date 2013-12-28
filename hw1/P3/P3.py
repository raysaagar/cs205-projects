from mrjob.job import MRJob
import math
import sys

class mrAnagram(MRJob):
	
	# mapper emits sorted string as key, and word
	def mapper(self, key, value):
		yield ''.join(sorted(list(value))), value

	# returns sorted string as key, plus length of list and word list
	def reducer(self, key, values):
		wordlist = list(values)
		yield key, [len(wordlist),wordlist]

if __name__ == '__main__':
	mrAnagram.run()
