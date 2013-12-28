from mrjob.job import MRJob
import math
import sys
import P6
import os

if __name__ == '__main__':

	# use os.listdir to get all txt files for input
        for files in os.listdir("."):
                if files.endswith(".txt"):
			# open input
			f = open(files,'r')
			# start job using open file
			mrJobInstance = P6.mrLetterCount()
			mrJobInstance.stdin = f
			runner = mrJobInstance.make_runner()
			runner.run()
			# push output into .out file
			f_out = open(files+".out",'w')
			for line in runner.stream_output():
				f_out.write(line)
			# close files for next job
			f_out.close()
			f.close()
