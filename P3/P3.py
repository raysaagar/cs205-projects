from mrjob.job import MRJob
import math
import sys


class mrAnagram(MRJob):

	def mapper(self, key, value):
		yield ''.join(sorted(list(value))), value

	def reducer(self, key, values):
		wordlist = list(values)
		yield key, [len(wordlist),wordlist]

if __name__ == '__main__':
	mrAnagram.run()
