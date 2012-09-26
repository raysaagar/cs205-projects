from mrjob.job import MRJob
import math
import sys
import P4

if __name__ == '__main__':
	converged = False
	
	init_input = open('graph.txt', 'r')

	while(not converged):
		mrJobInstance = P4.mrGraphAlg()
		mrJobInstance.stdin = init_input
		runner = mrJobInstance.make_runner()
		runner.run()
		
		counts = runner.counters()
		if (counts[0]['graph']['node'] == 0):
			converged = True
#		converged = True		
		f_out = open('out.txt','w')
		for line in runner.stream_output():
			f_out.write(line)
		f_out.close()
		init_input.close()
		init_input = open('out.txt','r')


