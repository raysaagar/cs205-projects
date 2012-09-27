from mrjob.job import MRJob
import math
import sys
import P6
import os

if __name__ == '__main__':

       # os.chdir("/mydir")
        for files in os.listdir("."):
                if files.endswith(".txt"):
			f = open(files,'r')
			mrJobInstance = P6.mrLetterCount()
			mrJobInstance.stdin = f
			runner = mrJobInstance.make_runner()
			runner.run()
			f_out = open(files+".out",'w')
			for line in runner.stream_output():
				f_out.write(line)
			f_out.close()
